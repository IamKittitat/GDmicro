def trans(merge_embedding_file, meta_file, embedding_vector_file):
    with open(meta_file, 'r') as fp:
        next(fp)  # Skip the header line
        samples = ['S' + line.split('\t')[0].strip() for line in fp]
    with open(merge_embedding_file, 'r') as f, open(embedding_vector_file, 'w') as o:
        for c, line in enumerate(f):
            ele = line.strip().split()
            te = [str(float(e)) for e in ele] 
            o.write(f"{samples[c]},{','.join(te)}\n")