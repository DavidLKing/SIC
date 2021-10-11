import sys
import pdb

scorer_output = open(sys.argv[1], 'r').readlines()
paras = open(sys.argv[2], 'r').readlines()

scores = [line for line in scorer_output if line.startswith('class')]

normalize = lambda x, x_max, x_min: (x - x_min) / (x_max - x_min)

logits = [float(x.split()[-1]) for x in scores]

log_max = max(logits)
log_min = min(logits)

norms = [normalize(x, log_max, log_min) for x in logits]

header = ['score', 'src', 'align', 'orig', 'para']

with open('scored.tsv', 'w') as outfile:
    
    outfile.write('\t'.join(header) + '\n')

    for score, para in zip(norms, paras):
        para = para.split(' [SEP] ')
        src_align = para[0].split(' [@] ')
        orig_para = para[1].split(' [@] ')
        src = src_align[0]
        align = src_align[1]
        orig = orig_para[0]
        para = orig_para[1]
        
        outfile.write('\t'.join([str(score), src, align, orig, para]))


pdb.set_trace()
