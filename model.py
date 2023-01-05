import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, n_groups, topk, group_y=None, n_labels=3993):
        super(Classifier, self).__init__()
        hidden_dim = 1000
        self.l1 = nn.Linear(5000, hidden_dim)
        # self.l2=nn.Linear(hidden_dim, n_labels)
        self.meta = nn.Linear(hidden_dim, n_groups)
        self.topk = topk
        self.group_y = group_y
        self.extreme = nn.Embedding(n_labels, n_groups)

    def get_candidates(self, group_logits):
        logits = group_logits.detach()
        scores, indices = torch.topk(logits, k=self.topk)
        candidates = []
        candidates_scores = []
        for index, score in zip(indices, scores):
            candidates.append(self.group_y[index])
            candidates_scores.append([np.full(c.shape, s) for c, s in zip(candidates[-1], score)])
            candidates[-1] = np.concatenate(candidates[-1])
            candidates_scores[-1] = np.concatenate(candidates_scores[-1])
        max_candidates = max([i.shape[0] for i in candidates])
        candidates = np.stack([np.pad(i, (0, max_candidates - i.shape[0]), mode='edge') for i in candidates])
        candidates_scores = np.stack(
            [np.pad(i, (0, max_candidates - i.shape[0]), mode='edge') for i in candidates_scores])
        return indices, candidates, candidates_scores

    def forward(self, x, candidates=None, labels=None, group_labels=None):
        y = self.l1(x)
        y = F.relu(y)
        y = self.meta(y)
        y = torch.sigmoid(y)

        l = labels.to(dtype=torch.bool)
        target_candidates = torch.masked_select(candidates, l).detach().cpu()
        target_candidates_num = l.sum(dim=1).detach().cpu()
        groups, candidates, group_candidates_scores = self.get_candidates(y)

        bs = 0
        new_labels = []
        for i, n in enumerate(target_candidates_num.numpy()):
            be = bs + n
            c = set(target_candidates[bs: be].numpy())
            c2 = candidates[i]
            new_labels.append(torch.tensor([1.0 if i in c else 0.0 for i in c2 ]))
            if len(c) != new_labels[-1].sum():
                s_c2 = set(c2)
                for cc in list(c):
                    if cc in s_c2:
                       continue
                    for j in range(new_labels[-1].shape[0]):
                        if new_labels[-1][j].item() != 1:
                            c2[j] = cc
                            new_labels[-1][j] = 1.0
                            break
            bs = be
            labels = torch.stack(new_labels)
            candidates, group_candidates_scores =  torch.LongTensor(candidates), torch.Tensor(group_candidates_scores)
            embed_weights = self.extreme(candidates)
            y = y.unsqueeze(-1)
            logits = torch.bmm(embed_weights, y).squeeze(-1)
            loss_fn =torch.nn.BCEWithLogitsLoss()
            loss=loss_fn(logits, labels) + loss_fn(y, group_labels)

        return logits, loss