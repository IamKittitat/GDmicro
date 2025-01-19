import re

def trans(pca_file,node_feature_dir,pre,meta_file):
    meta=open(meta_file,'r')
    line=meta.readline()
    d={}
    while True:
        line=meta.readline().strip()
        if not line:break
        ele=line.split('\t')
        if ele[3]=='healthy':
            d['S'+ele[0]]='Health'
        else:
            d['S'+ele[0]]=ele[3]
    pca=open(pca_file,'r')
    node_feature_file = open(node_feature_dir+'/'+pre+'_different_nf_value.txt','w+')
    while True:
        line=pca.readline().strip()
        if not line:break
        ele=re.split(',',line)
        name=ele[0]
        ele[0]=re.sub('S','',ele[0])
        node_feature_file.write('\t'.join(ele)+'\t'+d[name]+'\n')

