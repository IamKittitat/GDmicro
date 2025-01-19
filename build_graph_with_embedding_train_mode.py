import re
import os
import trans_embedding_vector
import preprocess_matrix_pca
import transform_matrix_anno
import numpy as np
import higra as hg
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

def check_trans_visualize_graph(meta_file,outgraph,build_graph_dir,pre,olog):
    G=nx.Graph()
    f=open(meta_file,'r')
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
    health=[]
    f22=open(outgraph,'r')
    o=open(build_graph_dir+'/'+pre+'_pca_knn_graph_final.txt','w+')
    while True:
        line=f22.readline().strip()
        if not line:break
        ele=line.split()
        o.write(re.sub('S','',ele[0])+'\t'+re.sub('S','',ele[1])+'\n')
        edge=(ele[0],ele[1])
        all_edges.append(edge)
        if not d[ele[0]]=='Health':
            if ele[0] not in disease:disease.append(ele[0])
        else:
            if ele[0] not in health:health.append(ele[0])
        if not d[ele[1]]=='Health':
            if ele[1] not in disease:disease.append(ele[1])
        else:
            if ele[1] not in health:health.append(ele[1])

    o.close()

    G.add_edges_from(all_edges)
    print('The number of edges of '+pre+' PCA KNN graph:',G.number_of_edges())
    olog.write('The number of edges of '+pre+' PCA KNN graph: '+str(G.number_of_edges())+'\n')
    print('Whether '+pre+' PCA KNN graph connected? ',nx.is_connected(G),'\n')
    olog.write('Whether '+pre+' PCA KNN graph connected? '+str(nx.is_connected(G))+'\n\n')
    plt.figure()
    color_map=[]
    for node in G:
        if node in disease:
            color_map.append('red')
        else:
            color_map.append('green')
    nx.draw(G,node_size=400,node_color=color_map,with_labels = True,font_size=8)
    
    for i in set(all_case):
        if i=='Health':
            plt.scatter([],[], c=['green'], label='{}'.format(i))
        else:
            plt.scatter([],[], c=['red'], label='{}'.format(i))
    plt.legend()
    plt.savefig(build_graph_dir+'/'+pre+'_pca_knn_graph_final.png',dpi=400)
    
def build_graph_given_matrix_with_knn(check1,check2,pca_file,meta_file,knn_nn,build_graph_dir,pre,build_log_file,rfile):
    if not os.path.exists(check1) and not os.path.exists(check2):
        return
    
    meta=open(meta_file,'r')
    d={} # Sample -> label
    line=meta.readline().strip()
    dname=''
    drname={}
    while True:
        line=meta.readline().strip()
        if not line:break
        ele=line.split()
        drname['S'+ele[0]]=ele[2]
        if not ele[3]=='healthy':
            d['S'+ele[0]]=ele[3]
            dname=ele[3]
        else:
            d['S'+ele[0]]='Health'
    pca=open(pca_file,'r')
    X=[]
    y=[]
    did2name={}
    count=0
    while True:
        line=pca.readline().strip()
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
    
    outgraph=build_graph_dir+'/'+pre+'_pca_knn_graph_ini.txt'

    drecord=defaultdict(lambda:{})
    o=open(outgraph,'w+')
    for i in range(len(sources)):
        o.write(did2name[sources[i]]+'\t'+did2name[targets[i]]+'\t'+str(edge_weights[i])+'\n')
        drecord[did2name[sources[i]]][did2name[targets[i]]]=str(edge_weights[i])
        drecord[did2name[targets[i]]][did2name[sources[i]]]=str(edge_weights[i])
    o.close()
    correct=0
    total=len(X)
    ot=open(rfile,'w+')
    ot.write('All_samples\tNeighbors\n')
    for r in drecord:
        cl=d[r]
        dn=0
        hn=0
        fl=''
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
        ot.write(drname[r]+'\t')
        tem=[]
        for e in drecord[r]:
            tem.append(drname[e]+':'+d[e]+':'+drecord[r][e])
        ot.write('\t'.join(tem)+'\n')
    print('The acc of '+pre+' knn graph: ',correct/total,correct,'/',total)
    build_log_file.write('The acc of '+pre+' knn graph: '+str(float(correct/total))+' '+str(correct)+'/'+str(total)+'\n')
    check_trans_visualize_graph(meta_file,outgraph,build_graph_dir,pre,build_log_file) 

def build_dir(indir):
    if not os.path.exists(indir):
        os.makedirs(indir)

def build(merge_embedding_file,meta_file,pre,output_dir,kneighbor,rfile):
   embedding_vector_dir = output_dir+'/P1_embedding_vector' 
   pca_res_dir = output_dir+'/P2_pca_res'
   build_graph_dir = output_dir+'/P3_build_graph'
   node_feature_dir = output_dir+'/P4_node_feature'

   build_dir(embedding_vector_dir)
   build_dir(pca_res_dir)
   build_dir(build_graph_dir)
   build_dir(node_feature_dir)

   # Combine meta file and merge_embedding file to P1_embedding_vector folder
   embedding_vector_file = embedding_vector_dir+'/embedding_vector.txt'
   trans_embedding_vector.trans(merge_embedding_file,meta_file,embedding_vector_file)

   j1='associate.pdf'
   j2='auc_run.txt'
   if not os.path.exists(j2):
       o=open(j2,'w+')
       o.close()
   
   preprocess_matrix_pca.run_pca(j1,j2,embedding_vector_file,meta_file,pre,pca_res_dir)
   build_log_file = open(output_dir+'/build_log.txt','w+')
   build_graph_given_matrix_with_knn(j1,j2,pca_res_dir+'/'+pre+'_matrix_ef_pca.csv',meta_file,kneighbor,build_graph_dir,pre,build_log_file,rfile)
   os.system('rm '+j2)
   transform_matrix_anno.trans(pca_res_dir+'/'+pre+'_matrix_ef_pca.csv',node_feature_dir,pre,meta_file)
