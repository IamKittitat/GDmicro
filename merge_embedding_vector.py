def merge_data(files, indices, ofile):
    """
    Merges data from multiple files based on their respective indices into a single output file.

    Args:
        files: List of input file paths.
        indices: List of index lists corresponding to each input file.
        ofile: Path to the output file.
    """
    def read_and_map(file_path, idx_list):
        with open(file_path, 'r') as f:
            return {idx_list[i]: line.strip() for i, line in enumerate(f)}

    # Read and map the data from each file
    d = {}
    for file, idx_list in zip(files, indices):
        d.update(read_and_map(file, idx_list))

    # Write the combined data to the output file
    with open(ofile, 'w') as o:
        total = sum(len(idx) for idx in indices)
        for i in range(total):
            o.write(d[i] + '\n')
