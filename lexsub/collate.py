#!/usr/bin/env python3

import sys
import pdb
import argparse
import logging
import pprint

import nltk
from nltk import word_tokenize
nltk.data.path.append("/scratch2/king/nltk_data")
# oracle = True

class collate:
    """
    This is a class to recombine generted paraphrases into their original dialogs
    """


    def __init__(self):
        # args
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '--input', help="input file (e.g. filtered.tsv)", required=True)
        parser.add_argument('-c', '--corrected', help="dialogs file, usually called 'corrected.tsv'", required=True)
        parser.add_argument('-o', '--output', help="main output file (e.g. filtered.tsv)", required=True)
        parser.add_argument('-d', '--missed', help="missed.tsv", required=True)
        # TODO why is it not taking *this* option?
        # parser.add_argument('-n', '--ngram_size', help="max ngram size (e.g. 1, 2, or 3)", required=True)
        self.args = parser.parse_args()
        # logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename='collated.log',level=logging.DEBUG)
        self.logger.info("Options:\n{}".format(pprint.pformat(self.args)))

    def combine(self, para_file, orig_file):
        """
        for new paraphrase line
            for every line in orig_file
                if the new line matches orig line
                    combine additional info and add to output
        output = 
            revamped dialogs
            reformated paraphrases without dailogs
        """
        # check on counts
        total = len(para_file)
        orig_total = len(orig_file)
        found_dialog = 0
        new_dailogs = []
        new_output = []
        no_match = []
        labels = set()
        count = 0
        ### THIS IS FOR WHEN LATER WE'RE LOOKING FOR LONELY PARAPHRASES WE DON'T HAVE TO WORD_TOKENIZE IN A LOOP
        orig_tok = []
        for orig_line in orig_file:
            count += 1
            if count % 100 == 0:
                print("On", count, "of", orig_total)
            new_dailogs.append(orig_line)
            orig_line = orig_line.split('\t')
            orig_sent = ' '.join(word_tokenize(orig_line[0]))
            if len(orig_line) > 1:
                orig_tok.append((orig_line[1], orig_sent))
            for para_line in para_file:
                para_line = para_line.split('\t')
                # TODO The below line is how we get label paraphrases. 
                # Currently we have to manually add this back
                # How can we make this another switch?
                # TODO Switch to yml config file
                # para_line[1] == orig_line[1] and \
                if len(para_line) > 2 and \
                   len(orig_line) > 2 and \
                   para_line[1] == orig_sent and \
                   para_line[0] == orig_line[1]:
                        dialog_line = [para_line[4], para_line[0]] + orig_line[2:]
                        # for getting labels into training
                        labels.add('\t'.join([orig_line[1], orig_line[1]] + orig_line[2:]))
                        new_output.append('\t'.join(dialog_line))
                        dialog_line = 'paraphrase ' + '\t'.join(dialog_line)
                        new_dailogs.append(dialog_line)
                        found_dialog += 1
        # TODO make a function out of this mess
        print("Now finding missing paraphrases")
        count = 0
        for para_line in para_file:
            found = False
            count += 1
            if count % 1000 == 0:
                print("On", count, "of", total)
            # try:
            para_line = para_line.split('\t')
            # except:
            #     pdb.set_trace()
            for orig_line in orig_tok:
                if len(para_line) > 2 and \
                   len(orig_line) == 2 and \
                   para_line[1] == orig_line[1] and \
                   para_line[0] == orig_line[0]:
                    found = True
                    # pdb.set_trace()
            if not found:
                # pdb.set_trace()
                no_match.append(para_line)
        # TODO lots to figure out
        # pdb.set_trace()
        # assert(len(new_output) == len(para_file))
        # assert(len(new_dailogs) == len(para_file) + len(orig_file))
        print("Found", found_dialog, "of", total, "paraphrases")
        print(len(labels), "labels found")
        # print("Found", len(no_match), "labels")
        # pdb.set_trace()
        for l in labels:
            new_dailogs.append('label ' + l)
        return new_dailogs, new_output


if __name__ == '__main__':
    print("Usage: python3 collate.py filtered.tsv correct.tsv")
    c = collate()
    para = open(c.args.input, 'r').readlines()
    corr = open(c.args.corrected, 'r').readlines()
    print("Successfully loaded", len(para), "paraphrases and", len(corr), "original dialog turns")
    new_dial, new_para = c.combine(para, corr)
    # if oracle:
    #     new_dial = c.combine_oracle(new_dial)
    # OUTPUT
    with open(c.args.output, 'w') as dial:
        for turn in new_dial:
            dial.write(turn)
    with open(c.args.missed, 'w') as outfile:
        for para in new_para:
            outfile.write(para)

