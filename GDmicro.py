import re
import os
import sys
import argparse

import run_GCN_train_mode
import run_GCN_test_mode
import GDmicro_preprocess

def load_var(inv,infile):
    return (1, infile) if os.path.exists(infile) else (0, inv)

def scan_input(indir, disease, unique_feature, mode='train'):
    if mode not in ['train', 'merge']:
        raise ValueError("Invalid mode. Choose either 'train' or 'merge'.")

    file_suffix = 'train' if mode == 'train' else 'merge'
    file_paths = {
        'node_norm': f'{indir}/{disease}_sp_{file_suffix}_norm_node.csv',
        'train_raw': f'{indir}/{disease}_{file_suffix}_sp_raw.csv',
        'node_raw': f'{indir}/{disease}_sp_{file_suffix}_raw_node.csv',
        'meta': f'{indir}/{disease}_meta.tsv',
        'train_norm': f'{indir}/{disease}_{file_suffix}_sp_norm.csv',
    }

    results = {}
    checks = []
    for key, path in file_paths.items():
        check, value = load_var('', path)
        results[key] = value
        checks.append(check)

    if sum(checks) != 5 and unique_feature == 0:
        print('Some input files are not provided, check please!')
        exit()

    pre_features = {}
    pre_features_dir = f'{indir}/pre_features'
    if os.path.exists(pre_features_dir):
        for filename in os.listdir(pre_features_dir):
            pre = re.split('_', filename)[0]
            pre = re.sub('Fold', '', pre)
            pre = int(pre)
            pre_features[pre] = f'{pre_features_dir}/{filename}'

    return (
        results['node_norm'],
        results['train_raw'],
        results['node_raw'],
        results['meta'],
        results['train_norm'],
        pre_features
    )

def main():
    usage="GDmicro - Use GCN and domain adaptation to predict disease based on microbiome data."
    parser=argparse.ArgumentParser(prog="GDmicro.py",description=usage)
    parser.add_argument('-i','--input_file',dest='input_file',type=str,help="The directory of the input csv file.")
    parser.add_argument('-t','--train_mode',dest='train_mode',type=str,help="If set to 1, then will apply k-fold cross validation to all input datasets. This mode can only be used when input datasets all have labels and set as \"train\" in input file.")
    #parser.add_argument('-v','--close_cv',dest='close_cv',type=str,help="If set to 1, will close the k-fold cross-validation and use all datasets for training. Only work when \"train mode\" is off (-t 0). (default: 0)")
    parser.add_argument('-d','--disease',dest='disease',type=str,help="The name of the disease.")
    parser.add_argument('-k','--kneighbor',dest='kneighbor',type=str,help="The number of neighborhoods in the knn graph. (default: 5)")
    parser.add_argument('-b','--batchsize',dest='bsize',type=str,help="The batch size during the training process. (default: 64)")
    parser.add_argument('-e','--apply_node',dest='anode',type=str,help="If set to 1, then will apply node importance calculation, which may take a long time. (default: not use).")
    parser.add_argument('-n','--node_num',dest='nnum',type=str,help="How many nodes will be output during the node importance calculation process. (default:20).")
    parser.add_argument('-f','--feature_num',dest='fnum',type=str,help="How many features (top x features) will be analyzed during the feature influence score calculation process. (default: x=10)")
    parser.add_argument('-c','--cvfold',dest='cvfold',type=str,help="The value of k in k-fold cross validation.  (default: 10)")
    parser.add_argument('-s','--randomseed',dest='rseed',type=str,help="The random seed used to reproduce the result.  (default: not use)")
    parser.add_argument('-a','--domain_adapt',dest='doadpt',type=str,help="Whether apply domain adaptation to the test dataset. If set to 0, then will use MLP rather than domain adaptation. (default: use)")
    parser.add_argument('-r','--run_fi',dest='rfi',type=str,help="Whether run feature importance calculation process. If set to 0, then will not calculate the feature importance and contribution score. (default: 1)")
    #parser.add_argument('-r','--reverse',dest='reverse',type=str,help="If set to 1, then will use functional data as node features, and compositional data to build edges. (default: 0)")
    #parser.add_argument('-v','--embed_vector_node',dest='vnode',type=str,help="If set to 1, then will apply domain adaptation network to node features, and use embedding vectors as nodes.. (default: 0)")
    #parser.add_argument('-u','--unique_feature',dest='uf',type=str,help="If set to 1, then will only use compositional data to build edges and as node features.")
    parser.add_argument('-o','--outdir',dest='outdir',type=str,help="Output directory of test results. (Default: GDmicro_res)")

    args = parser.parse_args()
    input_file = args.input_file
    train_mode = args.train_mode
    bsize = args.bsize
    run_feature_importance = args.rfi
    anode = args.anode
    disease = args.disease
    nnum = args.nnum
    fnum = args.fnum
    kneighbor = args.kneighbor
    cvfold = args.cvfold
    rseed=args.rseed
    doadpt=args.doadpt
    output_dir=args.outdir
    close_cv=0
    reverse = 0
    vnode = 0
    unique_feature = 1

    # Set default values
    bsize = int(bsize) if bsize else 64
    run_feature_importance = int(run_feature_importance) if run_feature_importance else 1
    anode = int(anode) if anode else 0
    nnum = int(nnum) if nnum else 20
    fnum = int(fnum) if fnum else 10
    kneighbor = int(kneighbor) if kneighbor else 5
    train_mode = int(train_mode) if train_mode else 0
    cvfold = int(cvfold) if cvfold else 10
    rseed = int(rseed) if rseed else 0
    doadpt = int(doadpt) if doadpt else 1
    output_dir = output_dir if output_dir else "GDmicro_res"

    indir, oin = GDmicro_preprocess.preprocess(input_file,train_mode,disease,output_dir)
    if train_mode == 0:
        node_norm,train_raw,node_raw,meta,train_norm,pre_features = scan_input(indir,disease,unique_feature, mode='test')
        run_GCN_test_mode.run(node_norm,train_raw,node_raw,meta,disease,output_dir,kneighbor,pre_features,rseed,cvfold,doadpt,train_norm,fnum,nnum,close_cv,anode,reverse,vnode,unique_feature,bsize,run_feature_importance,oin)
    else:
        # node_norm = (samples, idx+features+label)
        # train_raw = raw data without meta data
        # node_raw = (samples, idx+features+label) with normalized
        # meta = metadata
        # train_norm = normalized data without meta data
        node_norm,train_raw,node_raw,meta,train_norm,pre_features = scan_input(indir,disease,unique_feature, mode='train')
        run_GCN_train_mode.run(node_norm,train_raw,node_raw,meta,disease,output_dir,kneighbor,rseed,cvfold,train_norm,fnum,nnum,anode,run_feature_importance)

if __name__=="__main__":
    sys.exit(main())