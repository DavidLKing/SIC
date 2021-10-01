#!/usr/bin/env python3

import sys
import pickle as pkl
import pdb

if len(sys.argv) == 4:
    limit = int(sys.argv[3])
elif len(sys.argv) == 3:
    limit = 50
else:
    sys.exit("Usage\npython3 balance.py paraphrases.pkl output.tsv (20)")

paraphrases = pkl.load(open(sys.argv[1], 'rb'))

gen_para = open(sys.argv[2], 'r').readlines()
header = gen_para.pop(0)

print("Total paraphrases loaded", len(paraphrases))

print("options:", sys.argv)

outfile = open('filtered.tsv', 'w')
outfile.write(header)
# pdb.set_trace()

outgroup = []

# purge anything more than 50---or an arbitrary limit
purge = []
for para in paraphrases:
    if len(paraphrases[para]) >= limit:
        purge.append(para)
for p in purge:
    paraphrases.pop(p)

print("Starting the filter with limit", limit)

for line in gen_para:
    line = line.split('\t')
    label = line[0]
    # pdb.set_trace()
    # TODO verify that all labels are accounted for
    if label in paraphrases:
        if len(paraphrases[label]) <= limit:
                # be sure to add to the arrays so our limit is an actual limit
                paraphrases[label].append('\t'.join(line))
                outgroup.append('\t'.join(line))

for lines in outgroup:
    outfile.write(lines)
