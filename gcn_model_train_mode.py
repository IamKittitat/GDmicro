import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden = 32
dropout = 0.5
lr = 0.01 
weight_decay = 1e-5
fastmode = 'store_true'

def encode_onehot(labels):
    classes=sorted(list(set(labels)),reverse=True)
    classes_dict={c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot,classes_dict

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_data(mlp_or_not,graph,node_file,input_sample):
    ##### Load input sample info ######
    f=open(input_sample,'r')
    line=f.readline()
    train_id={}
    idx_train=[]
    idx_test=[]
    lid=0
    c=0
    tid2name={}
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split()    
        if ele[-1]=='train':
            train_id['S'+ele[1]]=''
            if int(ele[1])>lid:
                lid=int(ele[1])
            idx_train.append(c)
        else:
            idx_test.append(c)
        tid2name[c].append(ele[2])
        c+=1

    print('Loading {} dataset...'.format(graph+' plus '+node_file))
    idx_features_labels = np.genfromtxt("{}".format(node_file),dtype=np.dtype(str))
    features=idx_features_labels[:, 1:-1]
    features=features.astype(float)
    
    a=np.array(idx_features_labels[:, 1:-1])
    a=a.astype(float)

    features=np.array(features)
    labels,classes_dict = encode_onehot(idx_features_labels[:, -1])
    f1=features[idx_train]
    f2=features[idx_test]
    l1=labels[idx_train]
    l2=labels[idx_test]
    features=np.concatenate((f1, f2), axis=0)
    labels=np.concatenate((l1, l2), axis=0)
    features = sp.csr_matrix(features, dtype=np.float32)

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}".format(graph),dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #### identity matrix
    if mlp_or_not=='mlp':
        adj=sp.identity(len(labels)).toarray()
        adj=sp.csr_matrix(adj)
    else:
        adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_test = range(len(idx_train), len(labels))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    features_train=features[:len(idx_train)]
    labels_train=labels[:len(idx_train)]

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_test = torch.LongTensor(idx_test)
    return adj, features, labels, features_train,labels_train, idx_test,idx_train,classes_dict,tid2name

class GraphConvolution(Module):
    def __init__(self,in_features,out_features,bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias=Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self,input,adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    def __repr__(self):
        return self.__class__.__name__ + ' (' +str(self.in_features) + ' -> '+str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, hidden_layer, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, hidden_layer)
        self.gc2 = GraphConvolution(hidden_layer, nclass)
        self.dropout = dropout
    def forward(self, x, adj):
        x = torch.nn.functional.relu(self.gc1(x, adj))
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return torch.nn.functional.log_softmax(x, dim=1)

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = Parameter(torch.empty(size=(in_features, out_features)))
        self.a = Parameter(torch.empty(size=(2 * out_features, 1)))
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # Linear transformation
        
        N = Wh.size(0)
        
        # Compute attention scores
        a_input = torch.cat([Wh.repeat(1, N).view(N * N, -1), Wh.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        # Mask out unconnected nodes
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Apply softmax
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)
        
        # Apply attention weights
        h_prime = torch.matmul(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha = 0.2, nheads = 4):
        super(GAT, self).__init__()
        self.dropout = dropout

        # Multi-head attention layers
        self.attentions = nn.ModuleList([GraphAttentionLayer(nfeat, nhid, dropout, alpha, True) for _ in range(nheads)])
        
        # Output layer (single-head)
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout, alpha, False)
    
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  # Multi-head attention
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)  # Output layer
        return F.log_softmax(x, dim=1)

def accuracy(output,labels):
    preds=output.max(1)[1].type_as(labels)
    correct=preds.eq(labels).double()
    correct=correct.sum()
    return correct/len(labels)

def AUC(output,labels):
    output=torch.exp(output)
    a=output.data.numpy()
    preds=a[:,1]
    fpr,tpr,_ = metrics.roc_curve(np.array(labels),np.array(preds))
    auc=metrics.auc(fpr,tpr)
    return auc

# features = node_norm
def train(epoch,train_idx,val_idx,model,optimizer,features,adj,labels,
          result_detailed_file,max_val_auc,result_dir,fold,classes_dict,tid2name,record, save_val_results = False):
    t=time.time()
    model.train()
    optimizer.zero_grad()
    output=model(features,adj)
    loss_train=torch.nn.functional.nll_loss(output[train_idx], labels[train_idx])
    acc_train = accuracy(output[train_idx], labels[train_idx])
    auc_train=AUC(output[train_idx], labels[train_idx])
    loss_train.backward()
    optimizer.step()

    model.eval()
    output=model(features,adj)
    loss_val = torch.nn.functional.nll_loss(output[val_idx], labels[val_idx])
    acc_val = accuracy(output[val_idx], labels[val_idx])
    auc_val = AUC(output[val_idx], labels[val_idx])
    print('Epoch: {:04d}'.format(epoch+1),'loss_train: {:.4f}'.format(loss_train.item()),'acc_train: {:.4f}'.format(acc_train.item()),'loss_val: {:.4f}'.format(loss_val.item()),'acc_val: {:.4f}'.format(acc_val.item()),'time: {:.4f}s'.format(time.time() - t),'AUC_train: {:.4f}'.format(auc_train.item()),'AUC_val: {:.4f}'.format(auc_val.item()))
    result_detailed_file.write('Epoch: {:04d}'.format(epoch+1)+' loss_train: {:.4f}'.format(loss_train.item())+' acc_train: {:.4f}'.format(acc_train.item())+' loss_val: {:.4f}'.format(loss_val.item())+' acc_val: {:.4f}'.format(acc_val.item())+' time: {:.4f}s'.format(time.time() - t)+' AUC_train: {:.4f}'.format(auc_train.item())+' AUC_val: {:.4f}'.format(auc_val.item())+'\n')
    if save_val_results and auc_val>max_val_auc and record==1:
        o3=open(result_dir+'/sample_prob_fold'+str(fold)+'_val.txt','w+')
        output_res=torch.exp(output[val_idx]).data.numpy()
        
        c=0
        dt={}
        for n in classes_dict:
            if int(classes_dict[n][0])==1:
                dt[0]=n
            else:
                dt[1]=n
        for a in output_res:
            nt=labels[val_idx[c]].data.numpy()
            o3.write(tid2name[int(val_idx[c])]+'\t'+str(a[0])+'\t'+str(a[1])+'\t'+str(labels[val_idx[c]].data.numpy())+'\t'+str(dt[int(nt)])+'\n')
            c+=1
    
    return auc_train, auc_val, torch.exp(output).data.numpy()

def test(model,idx_test,features,adj,labels,result_detailed_file,max_test_auc,result_dir,fn,classes_dict,tid2name,record):
    model.eval()
    output=model(features,adj)
    loss_test=torch.nn.functional.nll_loss(output[idx_test], labels[idx_test])
    preds=output[idx_test].max(1)[1].type_as(labels[idx_test])
    acc_test=accuracy(output[idx_test],labels[idx_test])
    auc_test=AUC(output[idx_test], labels[idx_test])
    print(" | Test set results:","loss={:.4f}".format(loss_test.item()),"accuracy={:.4f}".format(acc_test.item()),"AUC={:.4f}".format(auc_test.item()))
    result_detailed_file.write(" | Test set results:"+"loss={:.4f}".format(loss_test.item())+" accuracy: {:.4f}".format(acc_test.item())+" AUC: {:.4f}".format(auc_test.item())+'\n')
    if auc_test>max_test_auc and record==1:
        o3=open(result_dir+'/sample_prob_fold'+str(fn)+'_test.txt','w+')
        output_res=torch.exp(output[idx_test])
        output_res=output_res.data.numpy()
        c=0
        dt={}
        for n in classes_dict:
            if int(classes_dict[n][0])==1:
                dt[0]=n
            else:
                dt[1]=n
        for a in output_res:
            nt=labels[idx_test[c]].data.numpy()
            o3.write(tid2name[int(idx_test[c])]+'\t'+str(a[0])+'\t'+str(a[1])+'\t'+str(labels[idx_test[c]].data.numpy())+'\t'+str(dt[int(nt)])+'\n')
            c+=1
    return auc_test

def run_GCN_test(mlp_or_not,epochs,graph,node_file,outfile1,outfile2,input_sample):
    adj,features,labels,features_train,labels_train,idx_test,idx_train,classes_dict,tid2name=load_data(mlp_or_not,graph,node_file,input_sample)
    splits=StratifiedKFold(n_splits=10,shuffle=True,random_state=1234)

    epochs = epochs
    o1=open(outfile1,'w+')
    fn=0
    for train_idx,val_idx in splits.split(np.array(features_train),np.array(labels_train)):
        o1.write('Fold {}'.format(fn+1)+'\n')
        model = GCN(nfeat=features.shape[1], hidden_layer=hidden, nclass=labels.max().item() + 1, dropout=dropout)
        optimizer = torch.optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)
        for epoch in range(epochs):
            train(epoch,train_idx,val_idx,model,optimizer,features,adj,labels,o1)
            test(model,idx_test,features,adj,labels,o1)
        fn+=1
    o1.close()


####### Species style
#run_GCN_test('gcn',500,'Graph_File_test_last_raw_Sp/sp_pca_knn_graph_final_trans_USA.txt','Node_File/species_node_feature.txt','Res_record_Sp/r1_USA_sp_lasso_gcn.txt','Res_record_Sp/r2_USA_sp_lasso_gcn.txt','sample_USA_new.txt')
#run_GCN_test('gcn',500,'Graph_File_test_last_raw_Sp/sp_pca_knn_graph_final_trans_AUS.txt','Node_File/species_node_feature.txt','Res_record_Sp/r1_AUS_sp_lasso_gcn.txt','Res_record_Sp/r2_AUS_sp_lasso_gcn.txt','sample_AUS_new.txt')
#run_GCN_test('gcn',500,'Graph_File_test_last_raw_Sp/sp_pca_knn_graph_final_trans_China.txt','Node_File/species_node_feature.txt','Res_record_Sp/r1_China_sp_lasso_gcn.txt','Res_record_Sp/r2_China_sp_lasso_gcn.txt','sample_China_new.txt')
#run_GCN_test('gcn',500,'Graph_File_test_last_raw_Sp/sp_pca_knn_graph_final_trans_Denmark.txt','Node_File/species_node_feature.txt','Res_record_Sp/r1_Denmark_sp_lasso_gcn.txt','Res_record_Sp/r2_Denmark_sp_lasso_gcn.txt','sample_Denmark_new.txt')
#run_GCN_test('gcn',500,'Graph_File_test_last_raw_Sp/sp_pca_knn_graph_final_trans_French.txt','Node_File/species_node_feature.txt','Res_record_Sp/r1_French_sp_lasso_gcn.txt','Res_record_Sp/r2_French_sp_lasso_gcn.txt','sample_French_new.txt')

### eggNOG style
#run_GCN_test('gcn',500,'Graph_File_test_last_raw/eggNOG_pca_knn_graph_final_trans_Denmark.txt','Node_File/species_node_feature.txt','Res_record_retest_ECE/r1_Denmark_eggNOG_lasso_raw.txt','Res_record_retest_ECE/r2_Denmark_eggNOG_lasso_raw.txt','sample_Denmark_new.txt')



