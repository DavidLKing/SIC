import sys
import pdb

corr1 = open(sys.argv[1], 'r').readlines()
corr2 = open(sys.argv[2], 'r').readlines()
outfile = open(sys.argv[3], 'w')

def sep_paras(corr):
    output = []
    labels = []
    source = ()
    for line in corr:
        if line.startswith('label '):
            labels.append(line)
        elif not line.startswith('paraphrase '):
            output.append(source)
            source = (line, [])
        elif line.startswith('paraphrase '):
            assert(line.startswith('paraphrase '))
            source[1].append(line)
        elif line.startswith('label-paraphrase '):
            continue
    output.append(source)
    output.pop(0)
    return output, labels

corr1, labs1 = sep_paras(corr1)
corr2, labs2 = sep_paras(corr2)

assert(len(corr1) == len(corr2))
for line1, line2 in zip(corr1, corr2):
    assert(line1[0] == line2[0])
    outfile.write(line1[0])
    if line1[1] != line2[1]:
        paras = line1[1] + line2[1]
        for p in paras:
            outfile.write(p)
# if labs1 != labs2:
#     pdb.set_trace()
labs = labs1 + labs2
for label in labs:
    outfile.write(label)

