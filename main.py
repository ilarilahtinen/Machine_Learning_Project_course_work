import os
import torch
from sklearn.preprocessing import normalize
import numpy as np
from EurlexDataset import EurLexDataset
from scipy.sparse import csr_matrix, csc_matrix
from torch.utils.data import DataLoader
from model import Classifier
import torch.optim as optim
import torch.nn.functional as F


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    print(pred.shape)
    ret = []
    for k in topk:
        correct = (target * torch.zeros_like(target).scatter(1, pred[:, :k], 1)).float()
        ret.append(correct.sum() / k)
    return ret

def build_tree_by_level(dataset, eps: float, max_leaf: int, levels: list, groups_path):
    print('Clustering')
    x, y = dataset.data, dataset.label_ids
    print('Getting Labels Feature')
    labels_f = normalize(csr_matrix(y.T) @ csc_matrix(x))
    print(F'Start Clustering {levels}')
    levels, q = [2**x for x in levels], None
    for i in range(len(levels)-1, -1, -1):
        if os.path.exists(F'{groups_path}-Level-{i}.npy'):
            print(F'{groups_path}-Level-{i}.npy')
            labels_list = np.load(F'{groups_path}-Level-{i}.npy', allow_pickle=True)
            q = [(labels_i, labels_f[labels_i]) for labels_i in labels_list]
            break
    if q is None:
        q = [(np.arange(labels_f.shape[0]), labels_f)]
    while q:
        labels_list = [x[0] for x in q]
        assert sum(len(labels) for labels in labels_list) == labels_f.shape[0]
        if len(labels_list) in levels:
            level = levels.index(len(labels_list))
            print(F'Finish Clustering Level-{level}')
            np.save(F'{groups_path}-Level-{level}.npy', np.asarray(labels_list))
        else:
            print(F'Finish Clustering {len(labels_list)}')
        next_q = []
        for node_i, node_f in q:
            if len(node_i) > max_leaf:
                next_q += list(split_node(node_i, node_f, eps))

        q = next_q
    print('Finish Clustering')
    return labels_list


def split_node(labels_i: np.ndarray, labels_f: csr_matrix, eps: float):
    n = len(labels_i)
    c1, c2 = np.random.choice(np.arange(n), 2, replace=False)
    centers, old_dis, new_dis = labels_f[[c1, c2]].toarray(), -10000.0, -1.0
    l_labels_i, r_labels_i = None, None
    while new_dis - old_dis >= eps:
        dis = labels_f @ centers.T  # N, 2
        partition = np.argsort(dis[:, 1] - dis[:, 0])
        l_labels_i, r_labels_i = partition[:n//2], partition[n//2:]
        old_dis, new_dis = new_dis, (dis[l_labels_i, 0].sum() + dis[r_labels_i, 1].sum()) / n
        centers = normalize(np.asarray([np.squeeze(np.asarray(labels_f[l_labels_i].sum(axis=0))),
                                        np.squeeze(np.asarray(labels_f[r_labels_i].sum(axis=0)))]))
    return (labels_i[l_labels_i], labels_f[l_labels_i]), (labels_i[r_labels_i], labels_f[r_labels_i])

trainset=EurLexDataset("Eurlex/eurlex_train.txt")
testset=EurLexDataset("Eurlex/eurlex_test.txt")

groups = build_tree_by_level(trainset,1e-4, 100, [], './data/label_group')

new_group = []
for group in groups:
    new_group.append([i for i in group])
#np.save(f'./data/label_group.npy', np.array(new_group))
label_map={}
group_y_map=np.empty(3993)
for cluster_id in range(len(new_group)):
    for label_id in new_group[cluster_id]:
        label_map[label_id]=cluster_id
        group_y_map[label_id]=cluster_id



train_dataloader = DataLoader(trainset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(testset, shuffle=False)

model=Classifier(len(new_group),10 ).float()
optimizer=optim.Adam(model.parameters(), lr=0.01)
for i in range(5):
    for train_x, train_labels, train_Y in train_dataloader:
        optimizer.zero_grad()
        group_labels=group_y_map[train_labels]
        candidates=np.concatenate(new_group[group_labels],axis=0)
        output,loss=model(train_x.float(),labels=train_Y, candidates=candidates,group_labels=group_labels)
        loss.backward()
        optimizer.step()
model.eval()
with torch.no_grad():
    test_x, test_Y=next(iter(test_dataloader))
    output=model(test_x.float())
    loss=F.binary_cross_entropy(output, test_Y.float())
    acc=accuracy(output, test_Y, (1,3,5))
    print(acc)