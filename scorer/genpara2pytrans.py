import sys
import pdb
import pandas as pd

datas = pd.read_csv(sys.argv[1], sep='\t')

srcs = datas['src'].values.tolist()
aligns = datas['align'].values.tolist()
paras = datas['para'].values.tolist()
origs = datas['orig'].values.tolist()

with open(sys.argv[2], 'w') as outfile:
    for src, align, orig, para in zip(
        srcs, aligns, origs, paras
    ):
        outfile.write('1\t' + src + ' [@] ' + align + ' [SEP] ' + orig + ' [@] ' + para + '\n')

print("Done")