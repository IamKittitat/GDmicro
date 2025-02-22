import os
import trans_embedding_vector
import preprocess_matrix_pca
import transform_matrix_anno
import os
import numpy as np
import higra as hg
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

def check_trans_visualize_graph(sinfo,outgraph,out,pre,olog):
    G=nx.Graph()
    f=open(sinfo,'r')
    d={}
    line=f.readline()
    all_case=[]
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split()
        if not ele[3]=='healthy':
            d['S'+ele[0]]=ele[3]
        else:
            d['S'+ele[0]]='Health'
        all_case.append(ele[3])
    all_edges=[]

    disease=[]
    unknown=[]
    health=[]
    #print(outgraph)
    f22=open(outgraph,'r')
    o=open(out+'/'+pre+'_pca_knn_graph_final.txt','w+')
    while True:
        line=f22.readline().strip()
        if not line:break
        #print(line)
        ele=line.split()
        o.write(re.sub('S','',ele[0])+'\t'+re.sub('S','',ele[1])+'\n')
        edge=(ele[0],ele[1])
        all_edges.append(edge)
        if not d[ele[0]]=='Health':
            if not d[ele[0]]=='Unknown':
                if ele[0] not in disease:disease.append(ele[0])
            else:
                if ele[0] not in unknown:unknown.append(ele[0])
        else:
            if ele[0] not in health:health.append(ele[0])
        if not d[ele[1]]=='Health':
            if not d[ele[1]]=='Unknown':
                if ele[1] not in disease:disease.append(ele[1])
            else:
                if ele[1] not in unknown:unknown.append(ele[1])
        else:
            if ele[1] not in health:health.append(ele[1])

    #print(all_edges)
    #exit()
    o.close()

    G.add_edges_from(all_edges)
    print('The number of edges of '+pre+' PCA KNN graph:',G.number_of_edges())
    olog.write('The number of edges of '+pre+' PCA KNN graph: '+str(G.number_of_edges())+'\n')
    print('Whether '+pre+' PCA KNN graph connected? ',nx.is_connected(G),'\n')
    olog.write('Whether '+pre+' PCA KNN graph connected? '+str(nx.is_connected(G))+'\n\n')
    pos=nx.spring_layout(G,seed=3113794652)
    plt.figure()
    color_map=[]
    for node in G:
        if node in disease:
            color_map.append('red')
        elif node in health:
            color_map.append('green')
        elif node in unknown:
            color_map.append('gray')
    #print(len(color_map),len(G))
    #exit()
    nx.draw(G,node_size=400,node_color=color_map,with_labels = True,font_size=8)
    
    for i in set(all_case):
        if i=='healthy':
            plt.scatter([],[], c=['green'], label='{}'.format(i))
        elif i=='Unknown':
            plt.scatter([],[], c=['gray'], label='{}'.format(i))
        else:
            plt.scatter([],[], c=['red'], label='{}'.format(i))
    plt.legend()
    plt.savefig(out+'/'+pre+'_pca_knn_graph_final.png',dpi=400)
    
def build_graph_given_matrix_with_knn_construct_g(check1,check2,imatrix,sinfo,knn_nn,out,pre,olog,rfile):
    r=0
    if not os.path.exists(check1):
        r+=1
    if not os.path.exists(check2) :
        r+=1
    if r==2:
        return
    f1=open(sinfo,'r')
    d={} # Sample -> label
    line=f1.readline().strip()
    dname=''
    test_sample=[]
    wwl=1
    drname={}
    while True:
        line=f1.readline().strip()
        if not line:break
        ele=line.split()
        drname['S'+ele[0]]=ele[2]
        if not ele[3]=='healthy':
            d['S'+ele[0]]=ele[3]
            dname=ele[3]
        else:
            d['S'+ele[0]]='Health'
        if ele[5]=='test':
            test_sample.append('S'+ele[0])
            if ele[3]=='Unknown':
                wwl=0
    f2=open(imatrix,'r')
    X=[]
    y=[]
    did2name={}
    count=0
    while True:
        line=f2.readline().strip()
        if not line:break
        ele=re.split(',',line)
        y.append(d[ele[0]])
        tmp=[]
        for e in ele[1:]:
            tmp.append(float(e))
        X.append(tmp)
        did2name[count]=ele[0]
        count+=1
    X=np.array(X)
    graph,edge_weights=hg.make_graph_from_points(X, graph_type='knn',n_neighbors=knn_nn)
    sources, targets = graph.edge_list()
    #print(sources)
    #exit()
    
    outgraph=out+'/'+pre+'_pca_knn_graph_ini.txt'

    drecord=defaultdict(lambda:{})
    o=open(outgraph,'w+')
    for i in range(len(sources)):
        o.write(did2name[sources[i]]+'\t'+did2name[targets[i]]+'\t'+str(edge_weights[i])+'\n')
        drecord[did2name[sources[i]]][did2name[targets[i]]]=str(edge_weights[i])
        drecord[did2name[targets[i]]][did2name[sources[i]]]=str(edge_weights[i])
    o.close()
    #o.close()
    #exit()
    correct=0
    if wwl==1:
        total=len(X)
    else:
        total=len(X)-len(test_sample)
    ot=open(rfile,'w+')
    ot.write('All_samples\tNeighbors\n')
    for r in drecord:
        cl=d[r]
        dn=0
        hn=0
        fl=''
        if wwl==1:
            for e in drecord[r]:
                if d[e]=='Health':
                    hn+=1
                else:
                    dn+=1
                if hn>dn:
                    fl='Health'
            if dn>hn:
                fl=dname
            if cl==fl:
                correct+=1
            #if r in test_sample:
            if True:
                ot.write(drname[r]+'\t')
                tem=[]
                for e in drecord[r]:
                    tem.append(drname[e]+':'+d[e]+':'+drecord[r][e])
                ot.write('\t'.join(tem)+'\n')
        else:
            if r not in test_sample:
                for e in drecord[r]:
                    if d[e]=='Health':
                        hn+=1
                    else:
                        if not d[e]=="Unknown":
                            dn+=1
                if hn>dn:
                    fl='Health'
                if dn>hn:
                    fl=dname
                if cl==fl:
                    correct+=1
                ot.write(drname[r]+'\t')
                tem=[]
                for e in drecord[r]:
                    tem.append(drname[e]+':'+d[e]+':'+drecord[r][e])
                ot.write('\t'.join(tem)+'\n')
            else:
                ot.write(drname[r]+'\t')
                tem=[]
                for e in drecord[r]:
                    tem.append(drname[e]+':'+d[e]+':'+drecord[r][e])
                ot.write('\t'.join(tem)+'\n')

    print('The acc of '+pre+' knn graph: ',correct/total,correct,'/',total)
    olog.write('The acc of '+pre+' knn graph: '+str(float(correct/total))+' '+str(correct)+'/'+str(total)+'\n')
    check_trans_visualize_graph(sinfo,outgraph,out,pre,olog) 

def build(infile,insample,pre,odir,kneighbor,rfile):
   def build_dir(indir):
    if not os.path.exists(indir):
        os.makedirs(indir)
        
   p1=odir+'/P1_embedding_vector' 
   p2=odir+'/P2_pca_res'
   p3=odir+'/P3_build_graph'
   p4=odir+'/P4_node_feature'

   build_dir(p1)
   build_dir(p2)
   build_dir(p3)
   build_dir(p4)

   p1_out=p1+'/embedding_vector.txt'
   trans_embedding_vector.trans(infile,insample,p1_out)
   j1='associate.pdf'
   j2='auc_run.txt'
   if not os.path.exists(j2):
       o=open(j2,'w+')
       o.close()
   
   preprocess_matrix_pca.run_pca(j1,j2,p1_out,insample,pre,p2)
   o2=open(odir+'/build_log.txt','w+')
   build_graph_given_matrix_with_knn_construct_g(j1,j2,p2+'/'+pre+'_matrix_ef_pca.csv',insample,kneighbor,p3,pre,o2,rfile)
   if os.path.exists(j2):
    os.system('rm '+j2)
   transform_matrix_anno.trans(p2+'/'+pre+'_matrix_ef_pca.csv',p4,pre,insample)



#build('T2D_result/Graph_File/merge_embedding_Fold1.txt','../New_datasets/T2D_data_2012_Trans/T2D_meta.tsv','eggNOG','T2D_result/Graph_File/Fold1')
    
#build('../feature_eggNOG_AUS_new_embedding.txt','sample_AUS_new.txt','eggNOG','new_embedding_AUS_eggNOG')
#build('/home/heruiliao2/Deep_Learning_Project/New_methods_explore_20220403/MLP/Merge_Vector_Sp/merge_GER_sp_embedding.txt','sample_Denmark_new.txt','sp','Embedding_graph_sp_LOSO/new_embedding_GER_sp')
#build('../feature_out_test.txt','../../Graph_with_raw_data_from_paper_Merge_V2/EMG_LOO_test_China_128/sample_phenotype.txt','species','test_embedding')

    

