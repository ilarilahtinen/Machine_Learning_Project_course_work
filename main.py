from EurlexDataset import createDataFrame, EurLexDataSet
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Classifier
import argparse

def train(model,df_train,df_test,label_map,lr,epochs,max_len=3112, use_clustering=True,sampling="lightxml"):
    print("use clustering")
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
        train_data = EurLexDataSet(df_train, "train", label_map, candidates_num=max_len,sampling=args.sampling)
        test_data = EurLexDataSet(df_test, "test", label_map, candidates_num= max_len)

        trainloader = DataLoader(train_data, batch_size=16,
                                 shuffle=True)
        testloader = DataLoader(test_data, batch_size=16,
                                shuffle=False)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        train_loss =model.one_epoch(epoch,trainloader,optimizer,eval_loader=testloader,uniform_sampling=(args.sampling=="uniform"))

        ev_result = model.one_epoch(epoch, testloader, optimizer, mode='eval')

parser=argparse.ArgumentParser()

parser.add_argument('--use_clustering', action="store_true")
parser.add_argument('--no_clustering', dest="use_clustering", action="store_false")
parser.set_defaults(use_clustering=True)
parser.add_argument('--lr',type=float,required=False, default=0.001)
parser.add_argument('--epochs',type=int,required=False, default=25)
parser.add_argument('--hidden_dim',type=int,required=False, default=1000)
parser.add_argument('--candidate_topk',type=int,required=False, default=10)
parser.add_argument('--sampling',type=str,required=False, default="lightxml")
args=parser.parse_args()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("load dataset")
    df_train,df_test,label_map=createDataFrame()
    print(f'load  dataset with {len(df_train["input"])} train {len(df_test["input"])} test with {len(label_map)} labels done')
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
    if args.use_clustering:
        model=Classifier(len(label_map),clusters,hidden_dim=args.hidden_dim,candidates_topk=args.candidate_topk)
    else:
        model=Classifier(len(label_map),hidden_dim=args.hidden_dim,candidates_topk=args.candidate_topk)#,clusters)

    train(model,df_train,df_test,label_map,args.lr,args.epochs,use_clustering=args.use_clustering,sampling=args.sampling)




