import re
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import uuid

def trans_meta(in1,in2,out):
    f1=open(in1,'r')
    f2=open(in2,'r')
    o=open(out,'w+')
    line=f1.readline().strip()
    o.write(line+'\tclass\n')
    c=0
    while True:
        line=f1.readline().strip()
        if not line:break
        o.write(line+'\ttrain\n')
        c+=1 
    line=f2.readline()
    while True:
        line=f2.readline().strip()
        if not line:break
        ele=line.split('\t')
        ele[0]=str(c)
        o.write('\t'.join(ele)+'\ttest\n')
        c+=1
    o.close()

def trans_meta_train(input_file,out):
    f=open(input_file,'r')
    o=open(out,'w+')
    line=f.readline().strip()
    o.write(line+'\tclass\n')
    while True:
        line=f.readline().strip()
        if not line:break
        o.write(line+'\ttrain\n')
    o.close()

def extract_tout(tout,d,ofile):
    o=open(ofile,'w+')
    f=open(tout,'r')
    line=f.readline().strip()
    ele=line.split('\t')
    dc={}
    c=0
    arr=[]
    for e in ele:
        if e in d:
            dc[c]=''
            arr.append(e)
        c+=1
    o.write('\t'.join(arr)+'\n')
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        o.write(ele[0])
        c=0
        tem=[]
        for e in ele[1:]:
            if c in dc:
                tem.append(e)
            c+=1
        o.write('\t'+'\t'.join(tem)+'\n')
    o.close()


def normalize_data_small(input_file,mtype,meta,dtype,ofile,inmerge,inmerge_meta):
    f=open(meta,'r')
    meta_content=[]
    line=f.readline().strip()
    meta_content.append(line)
    c=0
    ag=0
    d={}
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        d[ele[2]]=''
        meta_content.append(line)
        label=line.split('\t')[3]
        if label=='Unknown':
            ag=1
        c+=1
    uid = uuid.uuid1().hex
    if ag==1:
        n_split_d=int(c/2)
        n_split_h=c-n_split_d
        ml_d=[dtype for i in range(n_split_d)]
        ml_h=['healthy' for i in range(n_split_h)]
        ml=ml_d+ml_h
        if len(meta_content[1:])<13:
            tmeta = 'tem_meta_' + uid + '.tsv'
            ot = open(tmeta, 'w+')
            ot.write(meta_content[0]+'\n')
            ft=open(inmerge_meta,'r')
            line=ft.readline()
            i=0
            while True:
                line=ft.readline().strip()
                if not line:break
                ele=line.split('\t')
                if ele[-1]=='train':
                    ot.write('\t'.join(ele[:-1])+'\n')
                else:
                    ele[3]=ml[i]
                    i+=1
                    ot.write('\t'.join(ele[:-1])+'\n')
            ot.close()
            tout='tem_matrix_' + uid + '.tsv'

            os.system('Rscript norm_features.R ' + mtype + ' ' + inmerge + ' ' + tmeta + ' ' + dtype + ' ' + tout)
            extract_tout(tout,d,ofile)
            os.system('rm ' + tmeta+' '+tout)
        else:
            tmeta='tem_meta_'+uid+'.tsv'
            ot=open(tmeta,'w+')
            ot.write(meta_content[0]+'\n')
            i=0
            for c in meta_content[1:]:
                ele=c.split('\t')
                ele[3]=ml[i]
                i+=1
                ot.write('\t'.join(ele)+'\n')
            ot.close()
            os.system('Rscript norm_features.R '+mtype+' '+input_file+' '+tmeta+' '+dtype+' '+ofile)
            os.system('rm '+tmeta)
    else:
        if len(meta_content[1:]) < 13:
            tout = 'tem_matrix_' + uid + '.tsv'
            os.system('Rscript norm_features.R ' + mtype + ' ' + inmerge + ' ' + inmerge_meta + ' ' + dtype + ' ' + tout)
            extract_tout(tout, d, ofile)
            os.system('rm ' + tout)
        else:
            os.system('Rscript norm_features.R '+mtype+' '+input_file+' '+meta+' '+dtype+' '+ofile)

def normalize_data(input_file,mtype,meta,dtype,ofile):
    f=open(meta,'r')
    meta_content=[]
    line=f.readline().strip()
    meta_content.append(line)
    c=0
    ag=0
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        #d[ele[2]]=''
        meta_content.append(line)
        label=line.split('\t')[3]
        if label=='Unknown':
            ag=1
        c+=1
    if ag==1:
        n_split_d=int(c/2)
        n_split_h=c-n_split_d
        ml_d=[dtype for i in range(n_split_d)]
        ml_h=['healthy' for i in range(n_split_h)]
        ml=ml_d+ml_h
        uid = uuid.uuid1().hex

        tmeta='tem_meta_'+uid+'.tsv'
        ot=open(tmeta,'w+')
        ot.write(meta_content[0]+'\n')
        i=0
        for c in meta_content[1:]:
            ele=c.split('\t')
            ele[3]=ml[i]
            i+=1
            ot.write('\t'.join(ele)+'\n')
        ot.close()
        #exit()
        os.system('Rscript norm_features.R '+mtype+' '+input_file+' '+tmeta+' '+dtype+' '+ofile)
        os.system('rm '+tmeta)
    else:
        os.system('Rscript norm_features.R '+mtype+' '+input_file+' '+meta+' '+dtype+' '+ofile)

def load_train_sp(input_file,d,all_sp):
    f=open(input_file,'r')
    samples=f.readline().strip().split()
    anno={}
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        c=0
        anno[ele[0]]=''
        for e in ele[1:]:
            d[ele[0]][samples[c]]=e
            c+=1
        all_sp[ele[0]]=''
    return samples,anno

def load_test_sp(input_file,d,anno):
    f=open(input_file,'r')
    samples=f.readline().strip().split()
    count=0
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        c=0
        if ele[0] in anno:
            count+=1
            for e in ele[1:]:
                d[ele[0]][samples[c]]=e
                c+=1
    #print(count,' species of training datasets are detected in test datasets.')
    return samples

def merge_sp(in1,in2,out):
    d=defaultdict(lambda:{})
    all_sp={}
    s1,anno=load_train_sp(in1,d,all_sp)
    s2=load_test_sp(in2,d,anno)
    samples=s1+s2
    o=open(out,'w+')
    o.write('\t'.join(samples)+'\n')
    for e in sorted(all_sp.keys()):
        o.write(e)
        for s in samples:
            if s in d[e]:
                o.write('\t'+str(d[e][s]))
            else:
                o.write('\t'+str(0))
        o.write('\n')

def load_item(input_file,d,all_item):
    f=open(input_file,'r')
    samples=f.readline().strip().split()
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split()
        c=0
        for e in ele[1:]:
            d[ele[0]][samples[c]]=e
            c+=1
        all_item[ele[0]]=''
    return samples
     
    
def merge_eggNOG(in1,in2,out):
    d=defaultdict(lambda:{})
    all_item={}
    s1=load_item(in1,d,all_item)
    s2=load_item(in2,d,all_item)
    samples=s1+s2
    o=open(out,'w+')
    o.write('\t'.join(samples)+'\n')
    for e in sorted(all_item.keys()):
        o.write(e)
        for s in samples:
            if s in d[e]:
                o.write('\t'+str(d[e][s]))
            else:
                o.write('\t'+str(0))
        o.write('\n')

def trans2node(input_file,meta,ofile):
    f=open(meta,'r')
    status=[]
    line=f.readline()
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t') 
        if ele[3]=='healthy':
            status.append('Health')
        else:
            status.append(ele[3])
    a=pd.read_table(input_file)
    a=a.T
    a=np.array(a)
    c=0
    o=open(ofile,'w+')
    for t in a:
        o.write(str(c))
        for v in t:
            o.write('\t'+str(v))
        o.write('\t'+status[c]+'\n')
        c+=1
    o.close()

def split_file(input_file,disease,output_dir):
    intrain=output_dir+'/Split_dir/Train'
    intest=output_dir+'/Split_dir/Test'
    if not os.path.exists(intrain):
        os.makedirs(intrain)
    if not os.path.exists(intest):
        os.makedirs(intest)

    f=open(input_file,'r')
    line=f.readline().strip()
    ele=re.split(',',line)
    sp=ele[4:]
    train_ab=[]
    test_ab=[]
    sample_train=[]
    sample_test=[]
    train_meta={}
    test_meta={}
    while True:
        line=f.readline().strip()
        if not line:break
        ele=re.split(',',line)
        if ele[1]=='train':
            sample_train.append(ele[0])
            train_meta[ele[0]]=[ele[2],ele[3]]
            train_ab.append(ele[4:])
        else:
            sample_test.append(ele[0])
            test_meta[ele[0]]=[ele[2],ele[3]]
            test_ab.append(ele[4:])
    o1=open(intrain+'/'+disease+'_meta.tsv','w+')
    o2=open(intrain+'/'+disease+'_sp_matrix.csv','w+')
    o1.write('sampleID\tstudyName\tsubjectID\tdisease\tcountry\n')
    c=0
    for s in sample_train:
        o1.write(str(c)+'\t'+train_meta[s][1]+'\t'+s+'\t'+train_meta[s][0]+'\t'+train_meta[s][1]+'\n')
        c+=1
    o2.write('\t'.join(sample_train)+'\n')
    c=0
    train_ab=np.array(train_ab)
    train_ab=train_ab.T
    for s in sp:
        tab=0
        for x in train_ab[c]:
            tab+=float(x)
        if tab==0:
            c+=1
            continue
        o2.write(s+'\t'+'\t'.join(train_ab[c])+'\n') 
        c+=1
    if len(sample_test)>0:
        o3=open(intest+'/'+disease+'_meta.tsv','w+')
        o4=open(intest+'/'+disease+'_sp_matrix.csv','w+')
        o3.write('sampleID\tstudyName\tsubjectID\tdisease\tcountry\n')
        c=0
        for s in sample_test:
            o3.write(str(c)+'\t'+test_meta[s][1]+'\t'+s+'\t'+test_meta[s][0]+'\t'+test_meta[s][1]+'\n')
            c+=1
        o4.write('\t'.join(sample_test)+'\n')
        c=0
        test_ab=np.array(test_ab)
        test_ab=test_ab.T
        for s in sp:
            tab=0
            for x in test_ab[c]:
                tab+=float(x)
            if tab==0:
                c+=1
                continue
            o4.write(s+'\t'+'\t'.join(test_ab[c])+'\n')
            c+=1
    return intrain,intest

def check_test_num(meta):
    f=open(meta,'r')
    line=f.readline()
    c=0
    while True:
        line=f.readline()
        if not line:break
        c+=1
    if c<13:
        return True
    else:
        return False

def pre_load(input_file):
    f=open(input_file,'r')
    d={}
    while True:
        line=f.readline().strip()
        if not line:break
        ele=re.split(',',line)
        d[ele[0]]=[ele[1],ele[2]]
    return d

def scan_test_num(input_file,disease):
    f=open(input_file,'r')
    arr=[]
    tn=0
    line=f.readline().strip()
    arr.append(line)
    d=pre_load('allmeta.tsv')
    oin=0
    while True:
        line=f.readline().strip()
        if not line:break
        ele=re.split(',',line)
        if ele[1]=='test':
            if ele[2]=='Unknown' and disease==d[ele[0]][1]:
                if ele[0] in d:
                    ele[2]=d[ele[0]][0]
                    oin=1
            #ele[2]='Unknown'
            tn+=1
        tem=','.join(ele)
        arr.append(tem)
    uid = uuid.uuid1().hex
    ninfile='inmatrix_'+uid+'.csv'
    o=open(ninfile,'w+')
    for a in arr:
        o.write(a+'\n')
    o.close()
    return ninfile,oin

def preprocess(input_file,train_mode,disease,output_dir):
    scan_res,oin=scan_test_num(input_file,disease)
    if not scan_res=='':
        input_file=scan_res
    intrain,intest=split_file(input_file,disease,output_dir)

    train_mode=train_mode
    dtype=disease
    out=output_dir+'/Preprocess_data'
    
    if not os.path.exists(out):
        os.makedirs(out)

    if not train_mode:
        train_mode=0
    else:
        train_mode=int(train_mode)

    intrain_meta=''
    intest_meta=''
    intrain_sp=''
    intest_sp=''
    for filename in os.listdir(intrain):
        if re.search('meta\.tsv',filename):
            intrain_meta=intrain+'/'+filename
        if re.search('sp_matrix',filename):
            intrain_sp=intrain+'/'+filename
    if train_mode==0:
        for filename in os.listdir(intest):
            if re.search('meta\.tsv',filename):
                intest_meta=intest+'/'+filename
            if re.search('sp_matrix',filename):
                intest_sp=intest+'/'+filename
    if train_mode==0:
        check_arr=[intrain_meta,intest_meta,intrain_sp,intest_sp]
    else:
        check_arr=[intrain_meta,intrain_sp]
    for i in check_arr:
        if i=='':
            print('Some required files are not provided. Please check!')
            exit()
        else:
            print("Load files -> "+i)
    if train_mode==0:
        print('Preprocess 1 - Merge metadata.')
        trans_meta(intrain_meta,intest_meta,out+"/"+dtype+'_meta.tsv')
        if check_test_num(intest_meta):
            merge_sp(intrain_sp,intest_sp,output_dir+"/Split_dir/Test/"+dtype+'_merge_sp_raw.csv')
            temerge=output_dir+"/Split_dir/Test/"+dtype+'_merge_sp_raw.csv'
            os.system('cp '+out+"/"+dtype+'_meta.tsv'+' '+output_dir+"/Split_dir/Test/"+dtype+'_meta_merge.tsv')
            temeta=output_dir+"/Split_dir/Test/"+dtype+'_meta_merge.tsv'
        print('Preprocess 2 - Normalize all abundance matrices.')
        normalize_data(intrain_sp,'species',intrain_meta,dtype,out+"/"+dtype+'_train_sp_norm.csv')
        if check_test_num(intest_meta):
            normalize_data_small(intest_sp, 'species', intest_meta, dtype, out + "/" + dtype + '_test_sp_norm.csv',temerge,temeta)
        else:
            normalize_data(intest_sp,'species',intest_meta,dtype,out+"/"+dtype+'_test_sp_norm.csv')
    
        print('Preprocess 3 - Merge training and test datasets.') 
        merge_sp(intrain_sp,intest_sp,out+"/"+dtype+'_sp_merge_raw.csv')
        merge_sp(out+"/"+dtype+'_train_sp_norm.csv',out+"/"+dtype+'_test_sp_norm.csv',out+"/"+dtype+'_sp_merge_norm.csv')
        print('Preprocess 4 - Convert combined matrices to node feature format.')
        trans2node(out+"/"+dtype+'_sp_merge_norm.csv',out+"/"+dtype+'_meta.tsv',out+"/"+dtype+'_sp_merge_norm_node.csv')
        trans2node(out+"/"+dtype+'_sp_merge_raw.csv',out+"/"+dtype+'_meta.tsv',out+"/"+dtype+'_sp_merge_raw_node.csv')
    else:
        print('Train mode - Preprocess 1 - Transform metadata.')
        trans_meta_train(intrain_meta,out+"/"+dtype+'_meta.tsv')
        print('Train mode - Preprocess 2 - Normalize all abundance matrices.')
        os.system('cp '+intrain_sp+' '+out+"/"+dtype+'_train_sp_raw.csv')
        normalize_data(intrain_sp,'species',intrain_meta,dtype,out+"/"+dtype+'_train_sp_norm.csv')
        print('Train mode - Preprocess 3 - Convert normalized matrices to node feature format.')
        trans2node(out+"/"+dtype+'_train_sp_norm.csv',out+"/"+dtype+'_meta.tsv',out+"/"+dtype+'_sp_train_norm_node.csv')
        trans2node(out+"/"+dtype+'_train_sp_raw.csv',out+"/"+dtype+'_meta.tsv',out+"/"+dtype+'_sp_train_raw_node.csv')
    if not scan_res=='':
        os.system('rm '+scan_res)
    return out,oin
