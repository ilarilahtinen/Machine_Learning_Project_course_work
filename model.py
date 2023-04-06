import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Classifier(nn.Module):
    #initialize model
    def __init__(self, n_labels, clusters=None, hidden_dim=1000, candidates_topk=10):
        super(Classifier, self).__init__()
        self.candidates_topk=candidates_topk

        self.clusters=clusters
        if self.clusters is not None:
            self.n_cluster_labels = len(clusters)
            self.l1 = nn.Linear(5000, hidden_dim)
            self.meta = nn.Linear(hidden_dim, self.n_cluster_labels)
            self.extreme =nn.Embedding(n_labels,hidden_dim)
            nn.init.xavier_uniform_(self.extreme.weight) 
        else:
            self.l1 = nn.Linear(5000, hidden_dim)
            self.meta = nn.Linear(hidden_dim, n_labels)
            self.emb= nn.Embedding(n_labels,n_labels)

    def get_candidates(self, cluster_logits, cluster_gd=None):
        logits=torch.sigmoid(cluster_logits.detach())
        if cluster_gd is not None:
            logits+=cluster_gd
        #get indexes for topk best performing cluster
        scores, indices =torch.topk(logits, k=self.candidates_topk)
        scores, indices = scores.cpu().detach().numpy(), indices.cpu().detach().numpy()
        #initialize variables
        candidates, candidates_scores =[], []
        #loop through variables
        for index, score in zip(indices, scores):
            # add labels from cluster to the candidates array
            candidates.append(self.clusters[index])
            # add scores to array
            candidates_scores.append([np.full(len(c), s) for c,s in zip(candidates[-1],score)])
            # as model process multiple samples simultaneously there are two -dim arrays which are now flatten
            candidates[-1]=np.concatenate(candidates[-1])
            candidates_scores[-1]=np.concatenate(candidates_scores[-1])
        max_candidates =max([i.shape[0] for i in candidates])
        # make all arrays equally long and stack them to numpy array
        candidates = np.stack([np.pad(i, (0, max_candidates - i.shape[0]), mode='edge') for i in candidates])
        candidates_scores = np.stack([np.pad(i, (0, max_candidates - i.shape[0]), mode='edge') for i in candidates_scores])
        # cluster indices, candidates label, cluster scores for each candidate label
        return indices, candidates, candidates_scores

    def forward(self, inputs, labels=None, cluster_labels=None, candidates=None,uniform_candidates=None):
        is_training=labels is not None
        inputs=inputs.to(torch.float32)
        #feed forward
        y=self.l1(inputs)
        hidden_output=F.relu(y)
        meta_output=self.meta(hidden_output)


        if self.clusters is None:
            # If there are no clusters either return loss and prediction
            # or only prediction
            if is_training:
                loss_fn= nn.BCEWithLogitsLoss()
                if uniform_candidates is not None:
                    emb_candidates=self.emb(uniform_candidates)
                    meta_output_emb=meta_output.unsqueeze(-1)
                    print(uniform_candidates.size())
                    print(emb_candidates.size())
                    print(meta_output_emb.size())
                    output=torch.bmm(emb_candidates,meta_output_emb).squeeze(-1)
                    loss= loss_fn(output,labels)
                    return y, loss
                loss = loss_fn(meta_output, labels)
                return y, loss
            else:
                return y
        if is_training:
            # change target labels from number format to boolean
            l=labels.to(dtype=torch.bool)
            # select label indexes which are correct 
            target_candidates = torch.masked_select(candidates, l).detach().cpu()
            target_candidates_num =l.sum(dim=1).detach().cpu()

        clusters, candidates, group_candidates_scores = self.get_candidates(meta_output,
                                                                          cluster_gd=cluster_labels if is_training else None)
        if is_training:
            b_start=0
            new_labels=[]
            for i, n in enumerate(target_candidates_num.numpy()):
                b_end=b_start+n
                target_candidate_set = set(target_candidates[b_start:b_end].numpy())
                pred_candidate=candidates[i]
                new_labels.append(torch.tensor([1.0 if ind in target_candidate_set else 0.0 for ind in pred_candidate]))
                # add labels from incorrect clusters if
                if len(target_candidate_set)!=new_labels[-1].sum():
                    pred_candidate_set=set(pred_candidate)
                    for c in target_candidate_set:
                        if c in pred_candidate_set:
                            continue
                        for j in range(new_labels[-1].shape[0]):
                            if new_labels[-1][j].item() !=1:
                                pred_candidate[j]=c
                                new_labels[-1][j]=1.0
                                break
                b_start=b_end
            labels=torch.stack(new_labels).cuda()
        candidates, group_candidates_scores=  torch.LongTensor(candidates).cuda(), torch.Tensor(group_candidates_scores).cuda()
        embed_weights = self.extreme(candidates)
        hidden_output_emb=hidden_output.unsqueeze(-1)
        logits = torch.bmm(embed_weights, hidden_output_emb).squeeze(-1)
        if is_training:
            loss_fn = torch.nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)

            loss+= loss_fn(meta_output, cluster_labels)
            return logits, loss
        else:
            candidates_scores = torch.sigmoid(logits)
            candidates_scores = candidates_scores * group_candidates_scores
            return y, candidates, candidates_scores

    def get_accuracy(self, candidates, logits, labels):
        if candidates is not None:
            candidates = candidates.detach().cpu()

        scores, indices = torch.topk(logits.detach().cpu(), k=10)

        acc1, acc3, acc5, total = 0, 0, 0, 0
        for i, l in enumerate(labels):
            l = set(np.nonzero(l)[0])

            if candidates is not None:
                labels = candidates[i][indices[i]].numpy()
            else:
                labels = indices[i, :5].numpy()

            acc1 += len(set([labels[0]]) & l)
            acc3 += len(set(labels[:3]) & l)
            acc5 += len(set(labels[:5]) & l)
            total += 1

        return total, acc1, acc3, acc5

    def one_epoch(self, epoch, dataloader, optimizer, mode="train", eval_loader=None, eval_step=20000, log=None,uniform_sampling=False):
        bar = tqdm.tqdm(total=len(dataloader))
        precision1, precision3, precision5 = 0, 0, 0
        g_precision1, g_precision3, g_precision5 = 0, 0, 0
        total, acc1, acc3, acc5 = 0, 0, 0, 0
        g_acc1, g_acc3, g_acc5 = 0, 0, 0
        train_loss = 0

        if mode == 'train':
            self.train()
        else:
            self.eval()

        pred_scores, pred_labels = [], []
        bar.set_description(f'{mode}-{epoch}')

        with torch.set_grad_enabled(mode=='train'):
            for step, data in enumerate(dataloader):
                batch = tuple(t for t in data)

                input_params={'inputs':batch[0].cuda()}

                if mode=='train':
                    input_params['labels']=batch[1].cuda()
                    if self.clusters is not None:
                        input_params['cluster_labels']=batch[2].cuda()
                        input_params['candidates']=batch[3].cuda()
                    elif uniform_sampling:
                        input_params['uniform_candidates']=batch[2].cuda()
                outputs=self(**input_params)

                bar.update(1)

                if mode == 'train':
                    loss = outputs[1]
                    train_loss=loss.item()


                    loss.backward()
                    optimizer.step()
                    self.zero_grad()

                    if step % eval_step == 0 and eval_loader is not None and step != 0:
                        results = self.one_epoch(epoch, eval_loader, optimizer, mode='eval')
                        precision1, precision3, precision5 = results[3:6]
                        g_precision1, g_precision3, g_precision5 = results[:3]
                        if self.group_y is not None:
                            log.log(f'{epoch:>2} {step:>6}: {precision1:.4f}, {precision3:.4f}, {precision5:.4f}'
                                    f' {g_precision1:.4f}, {g_precision3:.4f}, {g_precision5:.4f}')
                        else:
                            log.log(f'{epoch:>2} {step:>6}: {precision1:.4f}, {precision3:.4f}, {precision5:.4f}')

                    bar.set_postfix(loss=train_loss)
                elif self.clusters is None:
                    logits = outputs
                    if mode == 'eval':
                        labels = batch[1]
                        _total, _acc1, _acc3, _acc5 = self.get_accuracy(None, logits, labels.cpu().numpy())
                        total += _total;
                        acc1 += _acc1;
                        acc3 += _acc3;
                        acc5 += _acc5
                        precision1 = acc1 / total
                        precision3 = acc3 / total / 3
                        precision5 = acc5 / total / 5
                        bar.set_postfix(p1=precision1, p3=precision3, p5=precision5)
                    elif mode == 'test':
                        pred_scores.append(logits.detach().cpu())
                else:
                    group_logits, candidates, logits = outputs

                    if mode=="eval":
                        labels = batch[1]
                        cluster_labels=batch[2]

                        _total, _acc1, _acc3, _acc5 = self.get_accuracy(candidates, logits, labels.cpu().numpy())
                        total += _total
                        acc1 += _acc1
                        acc3 += _acc3
                        acc5 += _acc5
                        precision1 = acc1 / total
                        precision3 = acc3 / total / 3
                        precision5 = acc5 / total / 5

                        _, _g_acc1, _g_acc3, _g_acc5 = self.get_accuracy(None, group_logits, cluster_labels.cpu().numpy())
                        g_acc1 += _g_acc1;
                        g_acc3 += _g_acc3;
                        g_acc5 += _g_acc5
                        g_precision1 = g_acc1 / total
                        g_precision3 = g_acc3 / total / 3
                        g_precision5 = g_acc5 / total / 5
                        bar.set_postfix(p1=precision1, p3=precision3, p5=precision5, g_p1=g_precision1, g_p3=g_precision3, g_p5=g_precision5)
                    elif mode == 'test':
                        _scores, _indices = torch.topk(logits.detach().cpu(), k=100)
                        _labels = torch.stack([candidates[i][_indices[i]] for i in range(_indices.shape[0])], dim=0)
                        pred_scores.append(_scores.cpu())
                        pred_labels.append(_labels.cpu())

        if mode == 'eval':
            return g_precision1, g_precision3, g_precision5, precision1, precision3, precision5
        elif mode == 'test':
            return torch.cat(pred_scores, dim=0).numpy(), torch.cat(pred_labels, dim=0).numpy() if len(
                pred_labels) != 0 else None
        elif mode == 'train':
            return train_loss
