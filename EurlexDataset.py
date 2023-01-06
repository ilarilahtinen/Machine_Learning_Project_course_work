import os
import torch
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

from torch.utils.data import Dataset

import tqdm

def createDataFrame():
    labels = []
    label_map={}
    dataType=[]


    with open(f'Eurlex/eurlex_train.txt') as f:
        for i,l in enumerate(tqdm.tqdm(f)):
            if(i==0):
                params=l.split()
                rows = int(params[0])
                arg_count = int(params[1])
                label_count = int(params[2])
                inputs_train=np.zeros((rows,arg_count))
            else:
                text=l.replace('\n','')
                parts=text.split(" ",1)
                labels.append(parts[0])
                for label in parts[0].split(","):
                    if label == "":
                        continue
                    else:
                        label_map[int(label)] = 0
                for item in parts[1].split():
                    index,value=tuple(item.split(":"))
                    inputs_train[i-1,int(index)]=float(value)

    with open(f'Eurlex/eurlex_test.txt') as f:
        for i,l in enumerate(tqdm.tqdm(f)):
            if(i==0):
                params=l.split()
                rows = int(params[0])
                arg_count = int(params[1])
                label_count = int(params[2])
                inputs_test=np.zeros((rows,arg_count))
            else:
                text=l.replace('\n','')
                parts=text.split(" ",1)
                labels.append(parts[0])
                for label in parts[0].split(","):
                    if label=="":
                        continue
                    else:
                        label_map[int(label)]=0
                for item in parts[1].split():
                    index,value=tuple(item.split(":"))
                    inputs_test[i-1,int(index)]=float(value)


    df_train={"input":normalize(inputs_train), "label":labels}
    df_test = {"input": normalize(inputs_test), "label": labels}
    for i, k in enumerate(sorted(label_map.keys())):
        label_map[k]=i


    return df_train,df_test, label_map

class EurLexDataSet(Dataset):
    def __init__(self,df,mode,label_map, clusters=None, candidates_num=None):
        self.mode=mode
        self.df=df
        self.n_labels=3993
        self.label_map=label_map
        self.len=len(self.df["input"])

        if clusters is not None:
            self.candidates_num=candidates_num
            self.clusters=[]
            self.n_clusters_labels=len(clusters)
            self.map_clusters = np.empty(3993,dtype=np.int64)
            for index, labels in enumerate(clusters):
                self.clusters.append([])
                for label in labels:
                    self.clusters[-1].append(label_map[int(label)])
                self.map_clusters[self.clusters[-1]]=index
                self.clusters[-1]=np.array(self.clusters[-1])
            self.clusters=np.array(self.clusters, dtype=object)
        else:
            self.clusters=None
    def __getitem__(self, idx):
        input= self.df['input'][idx]
        labels = [self.label_map[int(i)] for i in self.df['label'][idx].split(",") if i!="" and int(i) in self.label_map]



        if self.clusters is not None:
            label_ids = torch.zeros(self.n_labels)
            label_ids= label_ids.scatter(0, torch.tensor(labels), torch.tensor([1.0 for i in labels]))

            cluster_labels = self.map_clusters[labels]

            cluster_labels_ids = torch.zeros(self.n_clusters_labels)
            cluster_labels_ids = cluster_labels_ids.scatter(0, torch.tensor(cluster_labels), torch.tensor([1.0 for i in cluster_labels]))

            candidates = np.concatenate(self.clusters[cluster_labels], axis=0)

            if len(candidates) < self.candidates_num:
                sample = np.random.randint(self.n_clusters_labels, size=self.candidates_num - len(candidates))
                candidates = np.concatenate([candidates, sample])
            elif len(candidates) > self.candidates_num:
                candidates = np.random.choice(candidates, self.candidates_num, replace=False)

            if self.mode =="train":
                return input, label_ids[candidates], cluster_labels_ids, candidates
            else:
                return input, label_ids, cluster_labels_ids, candidates

        labels_ids= torch.zeros(self.n_labels)
        labels_ids= labels_ids.scatter(0, torch.tensor(labels), torch.tensor([1.0 for i in labels]))

        return input, labels_ids

    def __len__(self):
        return self.len