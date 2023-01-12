import os
import tqdm
import joblib
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.preprocessing import normalize
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MultiLabelBinarizer

def get_sparse_feature(feature_file):
    x, labels=load_svmlight_file(feature_file, multilabel=True, zero_based=True, offset=1, n_features=5000)

    return normalize(x), labels


def build_tree_by_level(eps: float, max_leaf: int, levels:list, cluster_path):
    print("Clustering")
    sparse_x, sparse_labels = get_sparse_feature('Eurlex/eurlex_train.txt')
    mlb=MultiLabelBinarizer()
    sparse_y=mlb.fit_transform(sparse_labels)
    joblib.dump(mlb, cluster_path+"mlb")
    print("Getting Labels Feature")
    label_f = normalize(csr_matrix(sparse_y.T)@csc_matrix(sparse_x))
    print(F"Start Clustering {levels}")
    q=[(np.arange(label_f.shape[0]),label_f)]
    while q:
        labels_list = [x[0] for x in q]
        assert sum(len(labels) for labels in labels_list) == label_f.shape[0]
        if len(labels_list) in levels:
            level = levels.index(len(labels_list))
            print(F'Finish Clustering Level-{level}')
            np.save(F'{cluster_path}-Level-{level}.npy', np.asarray(labels_list))
        else:
            print(F'Finish Clustering {len(labels_list)}')
        next_q = []
        for node_i, node_f in q:
            if len(node_i) > max_leaf:
                next_q += list(split_node(node_i, node_f, eps))
            else:
                np.save(F'{cluster_path}-last.npy', np.array(labels_list, dtype=object))

        q = next_q
    print('Finish Clustering')
    return mlb


def split_node(labels_i: np.ndarray, labels_f: csr_matrix, eps: float):
    n = len(labels_i)
    c1, c2 = np.random.choice(np.arange(n), 2, replace=False)
    centers, old_dis, new_dis = labels_f[[c1, c2]].toarray(), -10000.0, -1.0
    l_labels_i, r_labels_i = None, None
    while new_dis - old_dis >= eps:
        dis = labels_f @ centers.T  # N, 2
        partition = np.argsort(dis[:, 1] - dis[:, 0])
        l_labels_i, r_labels_i = partition[:n // 2], partition[n // 2:]
        old_dis, new_dis = new_dis, (dis[l_labels_i, 0].sum() + dis[r_labels_i, 1].sum()) / n
        centers = normalize(np.asarray([np.squeeze(np.asarray(labels_f[l_labels_i].sum(axis=0))),
                                        np.squeeze(np.asarray(labels_f[r_labels_i].sum(axis=0)))]))
    return (labels_i[l_labels_i], labels_f[l_labels_i]), (labels_i[r_labels_i], labels_f[r_labels_i])





if __name__=='__main__':
    mlb=build_tree_by_level(1e-4, 100, [], './data')
    clusters=np.load(f'./data-last.npy', allow_pickle=True)
    new_clusters=[]
    for cluster in clusters:
        new_clusters.append([mlb.classes_[i] for i in cluster])
    np.save('./data/label_cluster.npy',np.array(new_clusters, dtype=object))
