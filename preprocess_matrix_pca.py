import re
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_ph(f):
    d={}
    line=f.readline()
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split()
        if ele[3]=='healthy':
            d['S'+ele[0]]='Health'
        else:
            d['S'+ele[0]]=ele[3]
    return d

def preprocess(embedding_vector_file,meta_file):
    f1=open(embedding_vector_file,'r')
    f2=open(meta_file,'r')
    d=load_ph(f2)
    X=[]
    y=[]
    samples=[]
    while True:
        line=f1.readline().strip()
        if not line:break
        ele=re.split(',',line)
        y.append(d[ele[0]])
        samples.append(ele[0])
        tem=[]
        for e in ele[1:]:
            tem.append(float(e))
        X.append(tem)
    X=np.array(X)
    scaler=StandardScaler()
    X=scaler.fit_transform(X)

    return X,y,samples

def pca(X,y,samples,omatrix):
    pca=PCA(n_components=0.95)
    reduced_x = pca.fit_transform(X)
    crc_x, crc_y, health_x, health_y = [], [], [], []
    for i in range(len(reduced_x)):
        x_value = reduced_x[i]
        x_0, x_1 = x_value[0], x_value[1]
        label = y[i]

        if label == 'Health':
            if len(x_value)==1:
                health_x.append(x_0)
                health_y.append(0)
            else:
                health_x.append(x_0)
                health_y.append(x_1)
        else:
            if len(x_value)==1:
                crc_x.append(x_0)
                crc_y.append(0)
            else:
                crc_x.append(x_0)
                crc_y.append(x_1)
            
    o=open(omatrix,'w+')
    i=0
    for s in samples:
        o.write(s)
        for e in reduced_x[i]:
            o.write(','+str(e))
        o.write('\n')
        i+=1
    o.close()

def run_pca(check1,check2,embedding_vector_file,meta_file,pre,out):
    if os.path.exists(check1) or os.path.exists(check2):
        X,y,samples=preprocess(embedding_vector_file,meta_file)
        pca(X,y,samples,out+'/'+pre+'_matrix_ef_pca.csv')