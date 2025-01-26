import re
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
import run_MLP_embedding_train_mode
from gcn_model_train_mode import encode_onehot,GCN,train,normalize,sparse_mx_to_torch_sparse_tensor
from calculate_avg_acc_of_cross_validation_train_mode import cal_acc_cv
import torch
from sklearn.metrics.pairwise import cosine_similarity
from random import sample
import random
import uuid
from numpy import savetxt

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True

def build_dir(inp):
    if not os.path.exists(inp):
        os.makedirs(inp)

def select_features(train_raw,node_raw,train_idx,feature_dir,meta_file,disease,fold_number):
    inmatrix=pd.read_table(train_raw)
    inmatrix=inmatrix.iloc[:,train_idx]
    inmatrix.to_csv("tem_e.tsv",sep="\t")
    f=open("tem_e.tsv",'r')
    o=open('tem_e2.tsv','w+')
    line=f.readline().strip()
    o.write(line+'\n')
    while True:
        line=f.readline().strip()
        if not line:break
        o.write(line+'\n')
    o.close()
    os.system('rm tem_e.tsv')
    fm=open(meta_file,'r')
    om=open('tem_meta.tsv','w+')
    line=fm.readline()
    om.write(line)
    c=0
    while True:
        line=fm.readline().strip()
        if not line:break
        if c in train_idx:
            om.write(line+'\n')
        c+=1
    om.close()
    print('Run the command: Rscript feature_select_model_nodirect.R tem_e2.tsv tem_meta.tsv eggNOG '+disease)
    os.system('Rscript feature_select_model_nodirect.R tem_e2.tsv tem_meta.tsv eggNOG '+disease)
    os.system('mv tem_meta.tsv '+feature_dir+'/meta_Fold'+str(fold_number)+'.tsv')
    f2=open('eggNOG_feature_weight.csv','r')
    line=f2.readline()
    d={}
    t=0
    while True:
        line=f2.readline().strip()
        if not line:break
        line=re.sub('\"','',line)
        ele=re.split(',',line)
        t+=1
        if ele[-1]=='NA':continue
        if float(ele[-1])==0:continue
        d[ele[0]]=''
    print(':: Log: There are '+str(len(d))+'/'+str(t)+' features selected!\n')
    os.system('mv eggNOG_feature_weight.csv '+feature_dir+'/eggNOG_feature_weight_Fold'+str(fold_number)+'.csv')
    os.system('mv eggNOG_evaluation.pdf '+feature_dir+'/eggNOG_evaluation_Fold'+str(fold_number)+'.pdf')
    f3=open(node_raw,'r')
    line=f3.readline()
    o2=open('tem_e3.tsv','w+')
    o2.write(line)
    while True:
        line=f3.readline().strip()
        if not line:break
        ele=line.split('\t')
        if ele[0] not in d:continue
        o2.write(line+'\n')
    o2.close()
    os.system('rm tem_e2.tsv')
    os.system('mv tem_e3.tsv '+feature_dir+'/eggNOG_features_Fold'+str(fold_number)+'.tsv')
    a=feature_dir+'/eggNOG_features_Fold'+str(fold_number)+'.tsv'
    #a=feature_dir+'/eggNOG_features_Fold'+str(fold_number)+'.tsv'
    return a

def hard_case_split(infeatures,inlabels):
    splits=StratifiedKFold(n_splits=10,shuffle=True,random_state=1234)
    dist=cosine_similarity(infeatures,infeatures)
    dist_abs=np.maximum(dist,-dist)
    did2d={} # ID -> Minimum distance
    species_id=0
    for s in dist_abs:
        res=np.argsort(s)[::-1]
        for r in res:
            if r==species_id:continue
            did2d[r]=s[r]
            break
        species_id+=1
    res=sorted(did2d.items(), key = lambda kv:(kv[1], kv[0])) 
    res_half=res[:int(len(res)/2)]
    candidate_crc=[]
    candidate_health=[]
    #clabels=[]
    for r in res_half:
        if inlabels[r[0]]=='CRC':
            candidate_crc.append(r[0])
        else:
            candidate_health.append(r[0])
        #clabels.append(inlabels[r[0]])
    #print(candidate,clabels)
    train_val_idx=[]

    for train_idx_sk,val_idx_sk in splits.split(infeatures,inlabels):
        val_num=len(val_idx_sk)
        #train_num=len(candidate)-val_num
        crc_num=int(val_num/2)
        health_num=val_num-crc_num
        vi1=sample(candidate_crc,crc_num)
        vi2=sample(candidate_health,health_num)
        vid=vi1+vi2
        tid=[]
        for i in range(len(infeatures)):
            if i in vid:continue
            tid.append(i)
        #print(tid,vid,len(tid),len(vid))
        #print(len(train_idx_sk),len(val_idx_sk))
        #exit()
        train_val_idx.append((tid,vid))



    #print(train_val_idx[:4])
    #print(len(train_val_idx))
    #exit()
    return train_val_idx

def avg_score(avc,vnsa):
    for s in avc:
        if vnsa[s]==0:
            avc[s]['Increase2Disease'] =0
            avc[s]['Increase2Health'] = 0
            avc[s]['Decrease2Disease'] = 0
            avc[s]['Decrease2Health'] = 0
        else:
            avc[s]['Increase2Disease']=sum(avc[s]['Increase2Disease'])/vnsa[s]
            avc[s]['Increase2Health']=sum(avc[s]['Increase2Health'])/vnsa[s]
            avc[s]['Decrease2Disease'] = sum(avc[s]['Decrease2Disease']) / vnsa[s]
            avc[s]['Decrease2Health'] = sum(avc[s]['Decrease2Health']) / vnsa[s]
    return avc

def iter_run(features,train_id,test_id , adj, labels, ot2, result_dir,classes_dict, idx_to_subjectId):
    model = GCN(nfeat=features.shape[1], hidden_layer=32, nclass=labels.max().item() + 1, dropout=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    max_train_auc = 0
    for epoch in range(150):
        train_auc, _, train_prob = train(epoch, np.array(train_id), np.array(test_id), model, optimizer, features, adj, labels, ot2, max_train_auc, result_dir, 0, classes_dict, idx_to_subjectId,  0)
        train_auc = float(train_auc)
        if train_auc > max_train_auc:
            max_train_auc = train_auc
            best_prob = train_prob

    return best_prob

def detect_dsp(graph, node_raw,feature_id, labels_raw,labels,adj, train_id, test_id, result_dir,ot2,classes_dict, idx_to_subjectId,species_id,sname,fold_number):
    setup_seed(10)
    dn={}
    idx_features_labels = np.genfromtxt("{}".format(node_raw), dtype=np.dtype(str))
    features = idx_features_labels[:, 1:-1]
    features = features.astype(float)
    features = np.array(features)
    features_raw=features.copy()
    features = sp.csr_matrix(features, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))
    feature_id = list(range(int(features.shape[1])))
    f=open(graph,'r')
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        ele[0]=int(ele[0])
        ele[1]=int(ele[1])
        if ele[0] not in dn:
            dn[ele[0]]={ele[1]:''}
        else:
            dn[ele[0]][ele[1]]=''
        if ele[1] not in dn:
            dn[ele[1]]={ele[0]:''}
        else:
            dn[ele[1]][ele[0]]=''
    tg=[] # only consider training data for now
    for s in dn:
        p=0
        n=0
        if s not in train_id:continue
        for s2 in dn[s]:
            if s2 not in train_id:continue
            if labels_raw[s2]=='Health':
                n+=1
            else:
                p+=1
        if p>=0 and n>=0:
            tg.append(s)
    print('There are '+str(len(tg))+' samples have both >=0 healthy and disease neighbors.')
    model = GCN(nfeat=features.shape[1], hidden_layer=32, nclass=labels.max().item() + 1, dropout=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    max_train_auc=0
    for epoch in range(150):
        print('DSD_Raw')
        train_auc, _, train_prob = train(epoch, np.array(train_id), np.array(test_id), model, optimizer, features,adj, labels, ot2, max_train_auc, result_dir, 0, classes_dict, idx_to_subjectId, 0)
        train_auc = float(train_auc)
        if train_auc > max_train_auc:
            max_train_auc = train_auc
            best_prob = train_prob

    tgc=[]
    for t in tg:
        if best_prob[t][0]>best_prob[t][1]:
            prl=0
        else:
            prl=1
        if prl==labels[t]:
            tgc.append(t)

    print(len(tgc),' samples will be used to detect driver species...')
    res={} # sample_id -> feature_id -> [0.55,-1,-1,0.52,0.33,-1,-1]
    arr=[] # sample_id list
    for t  in tgc:
        arr.append(t)
        raw_prob=best_prob[t][1]
        c_feature=features_raw.copy()
        t_feature=c_feature[t]
        ab_max=np.max(t_feature)
        ab_median=np.median(t_feature)
        ab_min=np.min(t_feature)
        res[t]={}
        for s in feature_id:
            if s not in species_id:continue
            res[t][s]=['-1','-1','-1','-1','-1','-1','-1']
            res[t][s][0]=str(raw_prob)
            raw_feature_value=t_feature[s]
            if float(raw_feature_value)==0:continue
            set_index=[]
            if float(raw_feature_value)==ab_min:
                features_one=features.clone().detach()
                features_one[t][s]=ab_median
                features_two=features.clone().detach()
                features_two[t][s]=ab_max
                set_index.append(5)
                set_index.append(6)
            elif float(raw_feature_value)==ab_max:
                features_one =features.clone().detach()
                features_one[t][s]=ab_median
                features_two =features.clone().detach()
                features_two[t][s]=ab_min
                set_index.append(1)
                set_index.append(2)
            else:
                features_one =features.clone().detach()
                features_one[t][s] = ab_max
                features_two =features.clone().detach()
                features_two[t][s] = ab_min
                set_index.append(3)
                set_index.append(4)


            bp1 = iter_run(features_one, train_id,test_id, adj, labels, ot2, result_dir,classes_dict, idx_to_subjectId)
            bp2 = iter_run(features_two, train_id, test_id, adj, labels, ot2, result_dir, classes_dict, idx_to_subjectId)
            res[t][s][set_index[0]]= str(bp1[t][1])
            res[t][s][set_index[1]] = str(bp2[t][1])
    health_lab=0
    for c in classes_dict:
        if c=='Health':
            if classes_dict['Health'][0]==1:
                health_lab=0
            else:
                health_lab = 1

    # Increase abundance (3, 5, 6) -> close to CRC or close to Health | Decrease abundance (1, 2, 4) -> close to CRC or close to Health
    # Raw: 0: raw_prob, Max: 1: Max2Median, 2: Max2Zero, Middle: 3: Middle2Max, 4: Middle2Zero, Zero: 5: Zero2Median, 6: Zero2Max
    # New rule: Raw: 0: raw_prob, Max: 1: Max2Median, 2: Max2Min, Middle: 3: Middle2Max, 4: Middle2Min, Min: 5: Min2Median, 6: Min2Max
    o=open(result_dir+'/driver_sp_stat_fold'+str(fold_number+1)+'.txt','w+')
    iab=[3,5,6]
    dab=[1,2,4]
    avc={} # Calculate average change of each feature across disease and healthy samples
            # feature_name-> Disease: change_value | Health: change_value
    sp_name = dict(zip(species_id, sname))

    o.write('Sample_ID\tLabel\t'+'\t'.join(sname)+'\n')

    # Calculate valid samples
    vsa={} # species_id -> valid sample
    vnsa={} # sname -> valid sample
    for t in res:
        #valid=0
        for s in feature_id:
            valid=0
            if s not in species_id: continue
            if s not in vsa:
                vsa[s]=0
                vnsa[sp_name[s]]=0
            if not res[t][s][1] == '-1':
                c1 = float(res[t][s][0]) - float(res[t][s][1])
                c2 = float(res[t][s][0]) - float(res[t][s][2])
                if c1 < 0 and c2 < 0 and abs(c1) < abs(c2):valid=1
                if c1 > 0 and c2 > 0 and c1 < c2:valid=1
            elif not res[t][s][3] == '-1':
                c3 = float(res[t][s][0]) - float(res[t][s][3])
                c4 = float(res[t][s][0]) - float(res[t][s][4])
                if c3 > 0 and c4 < 0:valid=1
                if c3 < 0 and c4 > 0:valid=1
            elif not res[t][s][5] == '-1':
                c5 = float(res[t][s][0]) - float(res[t][s][5])
                c6 = float(res[t][s][0]) - float(res[t][s][6])
                if c5 > 0 and c6 < 0 :valid=1
                if c5 < 0 and c6 > 0 :valid=1
            if valid==1:
                vsa[s]+=1
                vnsa[sp_name[s]]+=1


    for t in res:
        o.write(str(t)+'\t'+labels_raw[t]+'\t')
        tem=[]
        # tid=0
        for s in feature_id:
            if s not in species_id: continue
            tem.append(','.join(res[t][s]))
            if sp_name[s] not in avc:
                avc[sp_name[s]] = {'Increase2Disease': [], 'Increase2Health':[], 'Decrease2Disease': [],
                                   'Decrease2Health': []}

            if not res[t][s][1] == '-1':
                c1 = float(res[t][s][0]) - float(res[t][s][1])
                c2 = float(res[t][s][0]) - float(res[t][s][2])

                if health_lab == 1:
                    if c1 < 0 and c2<0 and abs(c1)<abs(c2):
                        avc[sp_name[s]]['Decrease2Health'].append(abs(c1)+abs(c2))
                    if c1>0 and c2>0 and c1<c2:
                        avc[sp_name[s]]['Decrease2Disease'].append(abs(c1)+abs(c2))

                else:
                    if c1 < 0 and c2<0 and abs(c1)<abs(c2):
                        avc[sp_name[s]]['Decrease2Disease'].append(abs(c1)+abs(c2))
                    if c1 > 0 and c2 > 0 and c1 < c2:
                        avc[sp_name[s]]['Decrease2Health'].append(abs(c1)+abs(c2))

            elif not res[t][s][3] == '-1':
                c3 = float(res[t][s][0]) - float(res[t][s][3])
                c4 = float(res[t][s][0]) - float(res[t][s][4])
                if vsa[s]<15:
                    if abs(c3) > 0.3 or abs(c4) > 0.3: continue
                if health_lab == 1:
                    if c3 >0 and c4<0:
                        avc[sp_name[s]]['Increase2Disease'].append(abs(c3))
                        avc[sp_name[s]]['Decrease2Health'].append(abs(c4))
                    if c3<0 and c4>0:
                        avc[sp_name[s]]['Increase2Health'].append(abs(c3))
                        avc[sp_name[s]]['Decrease2Disease'].append(abs(c4))
                else:
                    if c3 >0 and c4<0:
                        avc[sp_name[s]]['Increase2Health'].append(abs(c3))
                        avc[sp_name[s]]['Decrease2Disease'].append(abs(c4))
                    if c3 < 0 and c4 > 0:
                        avc[sp_name[s]]['Increase2Disease'].append(abs(c3))
                        avc[sp_name[s]]['Decrease2Health'].append(abs(c4))


            elif not res[t][s][5] == '-1':
                c5 = float(res[t][s][0]) - float(res[t][s][5])
                c6 = float(res[t][s][0]) - float(res[t][s][6])
                if vsa[s]<15:
                    if abs(c3) > 0.3 or abs(c4) > 0.3: continue
                if health_lab == 1:
                    if c5 > 0 and c6<0:
                        avc[sp_name[s]]['Increase2Disease'].append(abs(c5))
                        avc[sp_name[s]]['Decrease2Health'].append(abs(c6))
                    if c5 <0 and c6 > 0:
                        avc[sp_name[s]]['Decrease2Health'].append(abs(c5))
                        avc[sp_name[s]]['Decrease2Disease'].append(abs(c6))
                else:
                    if c5 > 0 and c6<0 and abs(c5)< abs(c6):
                        avc[sp_name[s]]['Increase2Health'].append(abs(c5))
                        avc[sp_name[s]]['Increase2Disease'].append(abs(c6))

                    if c5 < 0 and c6 > 0:
                        avc[sp_name[s]]['Increase2Disease'].append(abs(c5))
                        avc[sp_name[s]]['Increase2Health'].append(abs(c6))


        o.write('\t'.join(tem)+'\n')
    o.close()
    avc=avg_score(avc,vnsa)

    o2=open(result_dir+'/driver_sp_change_fold'+str(fold_number+1)+'.txt','w+')
    o2.write('Species_ID\tSpecies_name\tIncrease2Disease\tIncrease2Health\tDecrease2Disease\tDecrease2Health\tValid_s\n')
    c=1
    for s in sname:
        o2.write(str(c)+'\t'+s+'\t'+str(avc[s]['Increase2Disease'])+'\t'+str(avc[s]['Increase2Health'])+'\t'+str(avc[s]['Decrease2Disease'])+'\t'+str(avc[s]['Decrease2Health'])+'\t'+str(vnsa[s])+'\n')
        c+=1

def feature_importance_check(feature_id,train_idx,val_idx,features,adj,labels,result_dir,fold_number,classes_dict,idx_to_subjectId,feat_imp_fold_file,ot,species_name,fnum,feat_imp_local_file):
    setup_seed(10)
    cround=1
    top100={}
    selected = {}
    selected_arr = []
    while True:
        res={}
        prob_matrix = []
        if cround==2:break
        for i in feature_id:
            max_train_auc=0
            best_prob = []
            i=int(i)
            if i in selected:continue
            features_tem=[[x[i]] for x in features]
            features_tem=torch.Tensor(features_tem)
            model=GCN(nfeat=features_tem.shape[1], hidden_layer=32, nclass=labels.max().item() + 1, dropout=0.5)
            optimizer = torch.optim.Adam(model.parameters(),lr=0.01, weight_decay=1e-5)
            for epoch in range(50):
                train_auc, _, sample_prob = train(epoch,np.array(train_idx),np.array(val_idx),model,optimizer,features_tem,adj,labels,ot,max_train_auc,result_dir,fold_number+1,classes_dict,idx_to_subjectId,0)
                train_auc=float(train_auc)
                if train_auc>max_train_auc:
                    max_train_auc=train_auc
                    best_prob = sample_prob
            res[i]=float(max_train_auc)
            prob_matrix.append(best_prob[:, 1])
        res2=sorted(res.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)
        species_id=1
        if cround==1:
            for r in res2:
                feat_imp_fold_file.write(str(species_id)+'\t'+str(species_name[r[0]])+'\t'+str(r[1])+'\n')
                if species_id<fnum+1:
                    top100[int(r[0])]=str(species_name[r[0]])
                species_id+=1
            feat_imp_fold_file.close()
        selected[res2[0][0]]=res2[0][1]
        selected_arr.append(res2[0][0])
        cround+=1
        prob_matrix = np.array(prob_matrix).T
        savetxt(feat_imp_local_file, prob_matrix, delimiter=',')
    species_id=sorted(list(top100.keys()))
    sname=[]
    for s in species_id:
        sname.append(top100[s])
    return species_id,sname

def node_importance_check(selected,selected_arr,tem_train_id,val_idx,features,adj,labels,result_dir,fold_number,classes_dict,idx_to_subjectId,o5,o6,ot2,nnum):
    cround=1
    while True:
        res={}
        if cround==nnum+1:break
        for i in tem_train_id:
            max_val_auc=0
            i=int(i)
            if i in selected:continue
            if i in val_idx:continue
            train_idx=selected_arr+[i]
            model=GCN(nfeat=features.shape[1], hidden_layer=32, nclass=labels.max().item() + 1, dropout=0.5)
            optimizer = torch.optim.Adam(model.parameters(),lr=0.01, weight_decay=1e-5)
            for epoch in range(50):
                _, val_auc, _ = train(epoch,np.array(train_idx),np.array(val_idx),model,optimizer,features,adj,labels,ot2,max_val_auc,result_dir,fold_number+1,classes_dict,idx_to_subjectId,0, save_val_results=True)
                val_auc=float(val_auc)
                if val_auc>max_val_auc:
                    max_val_auc=val_auc
            res[i]=float(max_val_auc)
        res2=sorted(res.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)
        species_id=1
        if cround==1:
            for r in res2:
                if species_id==nnum+1:break
                o5.write(str(species_id)+'\t'+str(idx_to_subjectId[r[0]])+'\t'+str(r[1])+'\n')
                species_id+=1
            o5.close()
        selected[res2[0][0]]=res2[0][1]
        selected_arr.append(res2[0][0])
        cround+=1
    species_id=1
    for r in selected_arr:
        o6.write(str(species_id)+'\t'+str(idx_to_subjectId[r])+'\t'+str(selected[r])+'\n')
        species_id+=1
    o6.close()

def trans_node(infile,meta_file,ofile):
    f=open(meta_file,'r')
    line=f.readline()
    arr=[]
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        arr.append(ele[3])
    a=pd.read_table(infile)
    a=np.array(a).T
    c=0
    o=open(ofile,'w+')
    for s in a:
        o.write(str(c))
        for e in s:
            o.write('\t'+str(e))
        o.write('\t'+arr[c]+'\n')
        c+=1
    o.close()

def load_species_name(train_norm):
    species_name = {}
    with open(train_norm, 'r') as f:
        next(f)  # Skip the header line
        for cs, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            ele = line.split('\t')
            species_name[cs] = ele[0]
    return species_name
 
def load_metadata(meta_file):
    idx_to_subjectId = {}
    test_idx = []
    train_id = 0

    with open(meta_file, 'r') as fm:
        next(fm)  
        for c, line in enumerate(fm):
            line = line.strip()
            if not line:
                continue

            ele = line.split()
            idx_to_subjectId[c] = ele[2]

            if ele[-1] == 'test':
                test_idx.append(c)
            if ele[-1] in ['train', 'test']:
                train_id = c

    train_id += 1
    test_idx = np.array(test_idx)
    return idx_to_subjectId, test_idx, train_id

def run(node_norm,train_raw,node_raw,meta_file,disease,out,kneighbor,rseed,cvfold,train_norm,fnum,nnum,anode,run_feature_importance):
    if rseed != 0:
        setup_seed(rseed)
    
    species_name = load_species_name(train_norm)

    idx_features_labels = np.genfromtxt("{}".format(node_norm),dtype=np.dtype(str))
    features=idx_features_labels[:, 1:-1] # Remove idx, labels
    features=features.astype(float)
    features=np.array(features)

    # Value: Health or {Disease Name (Ex. T2D)}
    labels_raw = idx_features_labels[:, -1]
    labels_raw = np.array(labels_raw)

    splits=StratifiedKFold(n_splits=cvfold,shuffle=True,random_state=1234)

    feature_dir=out+'/Feature_File'
    graph_dir=out+'/Graph_File'
    result_dir=out+'/Res_File'
    build_dir(feature_dir)
    build_dir(graph_dir)
    build_dir(result_dir)

    result_detailed_dir = result_dir+'/r1.txt'
    result_summary_dir = result_dir+'/r2.txt'

    idx_to_subjectId, test_idx, train_id = load_metadata(meta_file)
    
    result_detailed_file=open(result_detailed_dir,'w+')
    fold_number=0

    for train_idx,val_idx in splits.split(features[:train_id],labels_raw[:train_id]):
        result_detailed_file.write('Fold {}'.format(fold_number+1)+'\n')
        print('Fold {}'.format(fold_number+1)+', Train:'+str(len(train_idx))+' Test:'+str(len(val_idx)))

        # Get P3_build_graph graph_final (0,1 for edge connection)
        graph = run_MLP_embedding_train_mode.build_graph_mlp(train_norm,train_idx,val_idx,meta_file,disease,fold_number+1,graph_dir,kneighbor,rseed,result_dir)

        # Train and testing 
        labels,classes_dict = encode_onehot(labels_raw)
        features = sp.csr_matrix(features, dtype=np.float32)
        features=torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(labels)[1])

        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}".format(graph),dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]),dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        feature_id=list(range(int(features.shape[1])))
        tem_train_id=list(range(train_id))
 
        model=GCN(nfeat=features.shape[1], hidden_layer=32, nclass=labels.max().item() + 1, dropout=0.5)
        optimizer = torch.optim.Adam(model.parameters(),lr=0.01, weight_decay=1e-5)
        max_val_auc=0

        for epoch in range(150):
            _, val_auc, _ = train(epoch,train_idx,val_idx,model,optimizer,features,adj,labels,result_detailed_file,max_val_auc,result_dir,fold_number+1,classes_dict,idx_to_subjectId,1, save_val_results=True)
            if val_auc>max_val_auc:
                max_val_auc=val_auc

        ## Feature importance
        if run_feature_importance==1: 
            feat_imp_fold_file=open(result_dir+'/feature_importance_fold'+str(fold_number+1)+'.txt','w+')
            feat_imp_local_file = open(result_dir + '/feature_local_importance_fold' + str(fold_number + 1) + '.txt', 'w+')
            uid=uuid.uuid1().hex
            ot=open(uid+'.log','w+')
            species_id,sname=feature_importance_check(feature_id,train_idx,val_idx,features,adj,labels,result_dir,fold_number,classes_dict,idx_to_subjectId,feat_imp_fold_file,ot,species_name,fnum,feat_imp_local_file)
            ot.close()
            os.system('rm '+uid+'.log')

            uid=uuid.uuid1().hex
            ot2=open(uid+'.log','w+')
            detect_dsp(graph,node_raw,feature_id,labels_raw,labels,adj,train_idx,val_idx,result_dir,ot2,classes_dict,idx_to_subjectId,species_id,sname,fold_number)
            ot2.close()
            os.system('rm '+uid+'.log')
        
        ## Node importance
        if anode==1:
            selected={}
            selected_arr=[]
            o5=open(result_dir+'/node_importance_single_fold'+str(fold_number+1)+'.txt','w+')
            o6=open(result_dir+'/node_importance_combination_fold'+str(fold_number+1)+'.txt','w+')
            uid=uuid.uuid1().hex
            ot2=open(uid+'.log','w+')
            node_importance_check(selected,selected_arr,tem_train_id,val_idx,features,adj,labels,result_dir,fold_number,classes_dict,idx_to_subjectId,o5,o6,ot2,nnum)
            ot2.close()
            os.system('rm '+uid+'.log')

        fold_number+=1

    result_detailed_file.close()
    cal_acc_cv(result_detailed_dir,result_summary_dir)