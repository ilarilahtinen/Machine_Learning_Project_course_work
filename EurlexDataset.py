import os
import torch
import pickle

import numpy as np
from sklearn.preprocessing import normalize

from torch.utils.data import Dataset

import tqdm

def createDataFrame():
    labels_train = []
    labels_test = []
    label_map={}
    dataType=[]


    with open(f'Eurlex/eurlex_train.txt') as f:
        for i,l in enumerate(tqdm.tqdm(f)):
            # parse information from fist line
            if(i==0):
                params=l.split()
                rows = int(params[0])
                arg_count = int(params[1])
                label_count = int(params[2])
                inputs_train=np.zeros((rows,arg_count))
            # parse information from other lines
            else:
                # remove end of the line character
                text=l.replace('\n','')
                parts=text.split(" ",1)
                labels_train.append(parts[0])
                # Initialize labelmap to map labels from 
                # data to form where indexes are sequential
                if parts[0]!="":
                    for label in parts[0].split(","):
                        label_map[label] = 0

                for item in parts[1].split():
                    index,value=tuple(item.split(":"))
                    inputs_train[i-1,int(index)]=float(value)
    not_in_train=[]
    with open(f'Eurlex/eurlex_test.txt') as f:
        for i,l in enumerate(tqdm.tqdm(f)):
            # parse information from fist line
            if(i==0):
                params=l.split()
                rows = int(params[0])
                arg_count = int(params[1])
                label_count = int(params[2])
                inputs_test=np.zeros((rows,arg_count))
            # parse information from other lines
            else:
                # remove end of the line character
                text=l.replace('\n','')
                parts=text.split(" ",1)
                labels_test.append(parts[0])
                # Initialize labelmap to map labels from 
                # data to form where indexes are sequential
                if parts[0] != "":
                    for label in parts[0].split(","):
                        if label not in label_map:
                           not_in_train.append(label)
                        label_map[label]=0
                for item in parts[1].split():
                    index,value=tuple(item.split(":"))
                    inputs_test[i-1,int(index)]=float(value)

    print(not_in_train)
    df_train={"input":normalize(inputs_train), "label":labels_train}
    df_test = {"input": normalize(inputs_test), "label": labels_test}
    for i, k in enumerate(sorted(label_map.keys())):
        label_map[k]=i

    return df_train,df_test, label_map

class EurLexDataSet(Dataset):
    def __init__(self,df,mode,label_map, clusters=None, candidates_num=None, sampling="lightxml"):
        self.mode=mode
        self.sampling=sampling
        self.df=df
        self.n_labels=len(label_map)
        self.label_map=label_map
        self.sampling=sampling
        self.len=len(self.df["input"])
        self.candidates_num=candidates_num
        
        if clusters is not None:
            # amount of cluster caandidates
            self.clusters=[]
            # amount of clusters
            self.n_clusters_labels=len(clusters)
            # create map from label index to cluster index
            self.map_clusters = np.empty(self.n_labels,dtype=np.int64)
            for index, labels in enumerate(clusters):
                self.clusters.append([])
                for label in labels:

                    self.clusters[-1].append(label_map[str(int(label))])
                self.map_clusters[self.clusters[-1]]=index
                self.clusters[-1]=np.array(self.clusters[-1])
            self.clusters=np.array(self.clusters, dtype=object)
        else:
            self.clusters=None
    def __getitem__(self, idx):
        inputs= self.df['input'][idx]
        # map labels to sequential form
        labels = [self.label_map[i] for i in self.df['label'][idx].split(",") if i in self.label_map and i!=""]
        if self.clusters is not None:
            # convert labels to binary form
            label_ids = torch.zeros(self.n_labels)
            label_ids= label_ids.scatter(0, torch.tensor(labels,dtype=torch.int64), torch.tensor([1.0 for i in labels]))
            # map label indexes to cluster indexes 
            cluster_labels = self.map_clusters[labels]
            # convert clusters to binary form
            cluster_labels_ids= torch.zeros(self.n_clusters_labels)
            cluster_labels_ids = cluster_labels_ids.scatter(0, torch.tensor(cluster_labels), torch.tensor([1.0 for i in cluster_labels]))
            if len(cluster_labels)>0:
                # candidates are labels which belongs to correct clusters
                candidates = np.concatenate(self.clusters[cluster_labels], axis=0)
            else:
                candidates= np.random.randint(self.n_clusters_labels, size=self.candidates_num )

            if len(candidates) < self.candidates_num:
                sample = np.random.randint(self.n_clusters_labels, size=self.candidates_num - len(candidates))
                candidates = np.concatenate([candidates, sample])
            elif len(candidates) > self.candidates_num:
                candidates = np.random.choice(candidates, self.candidates_num, replace=False)
            
            if self.mode =="train":
                # inputs, is label correct for candidates, is cluster correct, candidate labels
                return inputs, label_ids[candidates], cluster_labels_ids, candidates
            else:
                # inputs, is label correct for all labels, is cluster correct, candidate labels
                return inputs, label_ids, cluster_labels_ids, candidates

        labels_ids= torch.zeros(self.n_labels)
        labels_ids= labels_ids.scatter(0, torch.tensor(labels).to(torch.int64), torch.tensor([1.0 for i in labels]))
        if self.sampling=="uniform":
            negative_labels=np.arange(self.n_labels)[labels_ids==0]
            uniform_candidates=np.random.choice(negative_labels, self.candidates_num-len(labels), replace=False)
            return inputs, labels_ids,np.concatenate(uniform_candidates,labels)
        return inputs, labels_ids

    def __len__(self):
        return self.len
