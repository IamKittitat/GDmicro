import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import accuracy_score
from sklearn import metrics
import build_graph_with_embedding_train_mode
import random
from copy import deepcopy

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True

class MLPclassifica(nn.Module):
    def __init__(self,nfeat):
        super(MLPclassifica,self).__init__()

        self.hidden1=nn.Sequential(
            nn.Linear(
                in_features=nfeat,
                out_features=16,
                bias=True,
            ),
            nn.ReLU()
        )

        self.hidden2=nn.Sequential(
            nn.Linear(16,10),
            nn.ReLU()
        )

        self.classifica=nn.Sequential(
            nn.Linear(10,2),
            nn.Sigmoid()
        )
        self.dropout=0.5

    def forward(self,x):
        fc1=self.hidden1(x)
        self.featuremap=fc1.detach()
        fc2=self.hidden2(fc1)
        output=self.classifica(fc2)
        return fc1,fc2,output

def load_data(inmatrixf,meta_file,train_idx,val_idx,disease):
    train_idx=np.array(train_idx)
    val_idx=np.array(val_idx)

    inmatrix=pd.read_table(inmatrixf)
    inmatrix_train=inmatrix.iloc[:,train_idx]
    inmatrix_val=inmatrix.iloc[:,val_idx]

    inmeta=pd.read_table(meta_file)

    labels_train=inmeta.loc[train_idx,:]["disease"]
    labels_val=inmeta.loc[val_idx,:]["disease"]

    labels_train=labels_train.to_numpy()
    labels_train[labels_train==[disease]]=1
    labels_train[labels_train==['healthy']]=0
    
    labels_val=labels_val.to_numpy()
    labels_val[labels_val==[disease]]=1
    labels_val[labels_val==['healthy']]=0

    inmatrix_train=inmatrix_train.T
    inmatrix_val=inmatrix_val.T

    X_train=inmatrix_train.to_numpy()
    X_val=inmatrix_val.to_numpy()
    return X_train,X_val,labels_train,labels_val

def AUC(output,labels):
    a = output.data.numpy()
    preds = a[:,1]
    fpr,tpr,_ = metrics.roc_curve(np.array(labels),np.array(preds))
    auc = metrics.auc(fpr,tpr)
    return auc

#merge('T2D_result/Graph_File/feature_out_train_Fold1_eggNOG.txt','T2D_result/Graph_File/feature_out_test_Fold1_eggNOG.txt',list(range(340)),list(range(340,363)),'T2D_result/Graph_File/merge_embedding_Fold1.txt')
def merge_embedding_vector_train_mode(infile1, infile2, train_idx, val_idx, ofile):
    with open(infile1, 'r') as f1:
        train_data = {train_idx[i]: line.strip() for i, line in enumerate(f1)}
    with open(infile2, 'r') as f2:
        val_data = {val_idx[i]: line.strip() for i, line in enumerate(f2)}

    merged_data = {**train_data, **val_data}

    with open(ofile, 'w') as o:
        for idx in range(len(train_idx) + len(val_idx)):
            o.write(merged_data[idx] + '\n')
            
def build_graph_mlp(inmatrixf,train_idx,val_idx,meta_file,disease,fold_number,graph_dir,kneighbor,rseed,result_dir):
    if not rseed==0:
        setup_seed(rseed)
    train_res_stat_file = open(graph_dir+'/train_res_stat_Fold'+str(fold_number)+'.txt','w+')
    feature_out_train_dir = graph_dir+'/feature_out_train_Fold'+str(fold_number)+'_eggNOG.txt'
    feature_out_val_dir = graph_dir+'/feature_out_val_Fold'+str(fold_number)+'_eggNOG.txt'

    X_train,X_val,y_train,y_val=load_data(inmatrixf,meta_file,train_idx,val_idx,disease)    
    X_train_nots=torch.from_numpy(X_train.astype(np.float32))
    y_train_t=torch.from_numpy(y_train.astype(np.int64))
    X_val_nots=torch.from_numpy(X_val.astype(np.float32))
    y_val_t=torch.from_numpy(y_val.astype(np.int64))

    train_data_nots=Data.TensorDataset(X_train_nots,y_train_t)
    train_nots_loader=Data.DataLoader(
        dataset=train_data_nots,
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )

    max_test_acc=0
    max_test_auc=0
    for _ in range(10):
        best_auc=0
        mlpc_raw=MLPclassifica(nfeat=X_train.shape[1])
        optimizer=torch.optim.Adam(mlpc_raw.parameters(),lr=0.01,weight_decay=1e-5)
        loss_func=nn.CrossEntropyLoss()
        print_step=25

        for epoch in range(50):
            for step, (b_x, b_y) in enumerate(train_nots_loader):
                _,_,output=mlpc_raw(b_x)
                train_loss=loss_func(output,b_y)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                niter=epoch*len(train_nots_loader)+step+1

                _,_,output=mlpc_raw(X_val_nots)
                val_auc=AUC(output,y_val_t)
                _,pre_lab=torch.max(output,1)
                val_accuracy=accuracy_score(y_val_t,pre_lab)

                if float(val_auc)>float(best_auc):
                    best_auc=float(val_auc)
                    mlpc=deepcopy(mlpc_raw)

                if niter%print_step==0:
                    _,pre_lab=torch.max(output,1)
                    val_accuracy=accuracy_score(y_val_t,pre_lab)
                    val_auc=AUC(output,y_val_t)

        _,_,output=mlpc(X_train_nots)
        _,pre_lab=torch.max(output,1)
        feature_output=mlpc.featuremap.cpu()
        feature_out=np.array(feature_output)
        train_acc=accuracy_score(y_train_t,pre_lab)
        train_auc=AUC(output,y_train_t)
        print("train_accuracy:",train_acc,"train_AUC:",train_auc)
        _,_,output=mlpc(X_val_nots)
        _,pre_lab=torch.max(output,1)
        feature_output_val=mlpc.featuremap.cpu()
        feature_out_val=np.array(feature_output_val)
        val_accuracy=accuracy_score(y_val_t,pre_lab)
        val_auc=AUC(output,y_val_t)
        print("val_accuracy:",val_accuracy,"val_AUC:",val_auc)
        train_res_stat_file.write("Train accuracy: "+str(train_acc)+" Train AUC: "+str(train_auc)+"\nVal accuracy: "+str(val_accuracy)+" Val AUC: "+str(val_auc)+'\n')
        if val_auc>max_test_auc:
            max_test_auc=val_auc
            max_test_acc=val_accuracy
            np.savetxt(feature_out_train_dir,feature_out)
            np.savetxt(feature_out_val_dir,feature_out_val)
        if val_auc==max_test_auc and val_accuracy>max_test_acc:
            max_test_acc=val_accuracy
            max_test_auc=val_auc
            np.savetxt(feature_out_train_dir,feature_out)
            np.savetxt(feature_out_val_dir,feature_out_val)
    
    merge_embedding_vector_train_mode(feature_out_train_dir,feature_out_val_dir,train_idx,val_idx,graph_dir+'/merge_embedding_Fold'+str(fold_number)+'.txt')
    build_graph_with_embedding_train_mode.build(graph_dir+'/merge_embedding_Fold'+str(fold_number)+'.txt',meta_file,'eggNOG',graph_dir+'/Fold'+str(fold_number),kneighbor,result_dir+'/sample_kneighbors_all_fold'+str(fold_number)+'.txt')
    graph=graph_dir+'/Fold'+str(fold_number)+'/P3_build_graph/eggNOG_pca_knn_graph_final.txt'
    return graph