#!/usr/bin/env python3

import sys
import fasttext
import pdb
from tqdm import tqdm
import pandas as pd
from sapphire import Sapphire
from nltk.tokenize import word_tokenize

class Alignments():

    def __init__(self, binfile):
        model = fasttext.FastText.load_model(binfile)
        self.aligner = Sapphire(model)

    def align(self, sent_1, sent_2):
        # TODO maybe update tokenization, it seems
        # tokens_1 = word_tokenize(sent_1)
        # tokens_2 = word_tokenize(sent_2)
        tokens_1 = sent_1.split()
        tokens_2 = sent_2.split()
        word_align, phrase_align = self.aligner(tokens_1, tokens_2)
        if word_align != []:
            # depreciated format from foregone era
            # ###:### \t sent 1 \t file/label sent 2 \t file/label \t 1 \1 1 \t word alignments
            # probably should have put this part somewhere else
            prefix = '999:999\t'
            infix = '\tdia99.html#fake_label_1\t'
            suffix = '\tdia99.html#fake_label_2\t1\t1\t'
            outline = prefix + sent_1 + infix + sent_2 + suffix
            outline += ' '.join(['-'.join([str(x[0] - 1), str(x[1] - 1)]) for x in word_align])
            # TODO fix, hack for sarah's code
            outline += '\t\t\n'
            return outline, word_align, phrase_align
        else:
            return None, word_align, phrase_align


    def get_pairs_corrected(self, training_file):
        label_dict = {}
        for line in open(training_file, 'r'):
            if not line.startswith('#START'):
                line = line.strip().split('\t')
                query = line[0]
                label = line[1]
                if label not in label_dict:
                    label_dict[label] = []
                label_dict[label].append(query)
            for label in label_dict:
                label_dict[label] = sorted(set(label_dict[label]))
            pairs = []
            for label in label_dict:
                for ex1 in label_dict[label]:
                    for ex2 in label_dict[label]:
                        if ex1 != ex2:
                            pairs.append([ex1, ex2])
        return pairs

    def get_pairs(self, training_file):
        label_dict = {}
        datas = pd.read_csv(training_file)
        # pdb.set_trace()
        for example, label in zip(datas['query'].values.tolist(), datas['label'].values.tolist()):
            if label not in label_dict:
                label_dict[label] = []
            label_dict[label].append(example)
        for label in label_dict:
            label_dict[label] = sorted(set(label_dict[label]))
        pairs = []
        for label in label_dict:
            for ex1 in label_dict[label]:
                for ex2 in label_dict[label]:
                    if ex1 != ex2:
                        pairs.append([ex1, ex2])
        return pairs

if __name__ == '__main__':
    outfile = open('sapphire_alignments.tsv', 'w')
    binloc = '../../sapphire/model/wiki-news-300d-1M-subword.bin'
    align = Alignments(binloc)
    # align.align("This is a test", "this is not a test")
    # pairs = align.get_pairs('../data/vp_contextual_full/train_ctx.csv')
    pairs = align.get_pairs_corrected('../data/COSI/cosi_corrected.tsv')
    for pair in tqdm(pairs):
        line0, _, _ = align.align(pair[0], pair[1])
        if line0:
            line1, _, _ = align.align(pair[1], pair[0])
            outfile.write(line0)
            outfile.write(line1)
    outfile.close()
