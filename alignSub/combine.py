import sys
import pdb

old_file = open(sys.argv[1], 'r').readlines()
newfile = open(sys.argv[2], 'r').readlines()

newfile.pop(0)
# pdb.set_trace()
assert(len(newfile) == len(old_file))

def normalize(dist):
    max = 0
    min = 1
    for float_value in dist:
        if float_value > max:
            max = float_value
        elif float_value < min:
            min = float_value
    # normalize
    new_dist = []
    for old_score in dist:
        new_score = (old_score - min) / (max - min)
        # invert
        new_dist.append(new_score)
    # pdb.set_trace()
    return new_dist

dist = []
for line in newfile:
    line = line.strip().split('\t')
    dist.append(float(line[0]))

dist = normalize(dist)

outfile = open(sys.argv[3], 'w')
assert(len(old_file) == len(dist))
for line1, score in zip(old_file, dist):
    line1 = line1.strip().split('\t')
    newline = line1[0:-1] + [str(score)]
    outfile.write('\t'.join(newline) + '\n')
