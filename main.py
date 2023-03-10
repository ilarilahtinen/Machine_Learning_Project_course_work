from EurlexDataset import createDataFrame, EurLexDataSet
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Classifier

def train(model,df_train,df_test,label_map,max_len=3112, use_clustering=True):
    print(use_clustering)
    if use_clustering:
        clusters=np.load('./data/label_cluster.npy', allow_pickle=True)
        train_data=EurLexDataSet(df_train,"train",label_map,clusters, max_len)
        test_data = EurLexDataSet(df_test, "test", label_map, clusters, max_len)

        trainloader = DataLoader(train_data, batch_size=16,
                                 shuffle=True)
        testloader = DataLoader(test_data, batch_size=16,
                                shuffle=False)
    else:
        train_data = EurLexDataSet(df_train, "train", label_map, candidates_num=max_len,sampling="uniform")
        test_data = EurLexDataSet(df_test, "test", label_map, candidates_num= max_len)

        trainloader = DataLoader(train_data, batch_size=16,
                                 shuffle=True)
        testloader = DataLoader(test_data, batch_size=16,
                                shuffle=False)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(25):
        train_loss =model.one_epoch(epoch,trainloader,optimizer,eval_loader=testloader,uniform_sampling=True)

        ev_result = model.one_epoch(epoch, testloader, optimizer, mode='eval')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("load dataset")
    df_train,df_test,label_map=createDataFrame()
    print(f'load  dataset with '
          f'{len(df_train["input"])} train {len(df_test["input"])} test with {len(label_map)} labels done')
    clusters=np.load('./data/label_cluster.npy', allow_pickle=True)
    #train_data = EurLexDataSet(df_train, "train", label_map, clusters, 516)
    if clusters is not None:
        _clusters=[]
        for index, labels in enumerate(clusters):
            _clusters.append([])
            for label in labels:
                _clusters[-1].append(label_map[str(int(label))])
            _clusters[-1]=np.array(_clusters[-1])
        clusters=np.array(_clusters, dtype=object)
    #data=DataLoader(train_data)
    model=Classifier(len(label_map))#,clusters)

    train(model,df_train,df_test,label_map,use_clustering=False)




