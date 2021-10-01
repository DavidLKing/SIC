#!/usr/bin/env python3

import sys
import gensim
import pdb
import random
import pickle as pkl
import pandas as pd
from tqdm import tqdm
from patternPara import PatternSwap
from combineGold import Combine
from build_phrase_table import PhraseTable
from score import Score, lang_mod
from bert import BertBatch

# Usage: ./elmoEval.py elmoalignments.tsv ../data/.../batch*

class alignSub:

    def __init__(self):
        pass

    def align(self, src, tgt, indexes, phrase_table):
        # mocked up from PhraseTable.py to not create phrases
        src = src.split()
        tgt = tgt.split()
        # indexes = p.conv2range(indexes)
        for pair in indexes:
            # src_phrase = self.gen_phrase(src, pair[0])
            # tgt_phrase = self.gen_phrase(tgt, pair[1])
            src_phrase = src[pair[0]]
            tgt_phrase = tgt[pair[1]]
            if src_phrase != tgt_phrase:
                if src_phrase not in phrase_table:
                    phrase_table[src_phrase] = []
                if tgt_phrase not in phrase_table[src_phrase]:
                    phrase_table[src_phrase].append(tgt_phrase)
        return phrase_table

    def get_align(self, groups, phrases):
        for group in groups:
            src = group[0]
            tgt = group[1]
            idxes = group[2]
            if idxes != '':
                idxes = p.str2idx(idxes)
                phrases = p.align(src, tgt, idxes, phrases)
        return phrases

    def get_range_align(self, groups, phrases):
        for group in groups:
            src = group[0]
            tgt = group[1]
            idxes = group[2]
            if idxes != '':
                idxes = p.str2idx(idxes)
                # if use_phrase:
                idxes = p.conv2range(idxes)
                # else:
                #     # get analgous output to above
                #     idxes = [[[i[0]], [i[1]]] for i in idxes]
                phrases = p.align(src, tgt, idxes, phrases)
        return phrases

    def elmo_clean(self, elmos):
        new_elmos = []
        for line in elmos:
            if not line[2].startswith('No Align'):
                new_elmos.append(line)
        return new_elmos

    def rec_prec(self, den_dict, num_dict):
        num = 0
        den = 0
        for w1 in den_dict:
            for w2 in den_dict[w1]:
                if w1 in num_dict:
                    if w2 in num_dict:
                        num += 1
                den += 1
        return num / den

    def get_low_freq(self, corrected):
        labels = {}
        sents = []
        for line in corrected:
            line = line.strip().split('\t')
            if len(line) > 1:
                label = line[3]
                if label not in labels:
                    labels[label] = 0
                labels[label] += 1
                sents.append(line)
        return_sents = []
        for line in sents:
            if labels[line[3]] < 20:
                return_sents.append(line)
        return return_sents

    def get_labels(self, corrected):
        labels = set()
        for line in corrected:
            line = line.strip().split('\t')
            if len(line) > 1:
                label = line[3]
                labels.add(label)
        return labels

    def swap(self, sents, swap_dict):
        paraphrases = []
        mltplsrc = 0
        total = 0
        sum_multiple = 0
        sum_all = 0
        # header = [
        #     'swappable',
        #     'swap',
        #     'src',
        #     'align',
        #     'para',
        #     'orig',
        #     'label',
        #     'response',
        #     'cs guess',
        #     'cs correct',
        #     'color code 1',
        #     'color code 2'
        # ]
        # paraphrases.append(header)
        for line in sents:
            total += 1
            sent = line[0]
            for swappable in swap_dict:
                # TODO temp hack to stop history + i > we = hwestory
                # TODO there has to be a better way to do this: tokenization?
                if ' ' + swappable + ' ' in sent:
                    for swap in swap_dict[swappable]:
                        src = random.choice(swap_dict[swappable][swap]['src'])
                        align = random.choice(swap_dict[swappable][swap]['align'])
                        para = sent.replace(swappable, swap)
                        new_array = [swappable, swap, src, align, para] + line
                        try:
                            assert(len(new_array) == len(header) or len(new_array) == len(header) -2)
                        except:
                            pdb.set_trace()
                        paraphrases.append(new_array)
        return paraphrases

    def writeout(self, name, lines):
        with open(name, 'w') as of:
            for line in lines:
                # TODO we shouldn't need this check
                if line[0] != line[1]:
                    of.write('\t'.join(line[:7]) + '\n')

if __name__ == '__main__':

    # Simple Options

    '''
    For clarification:
        Phrasal = Sarah's variable based sentence templates
            lambda x y: [Do x like y] (you, hamsters) --> Do you like hamsters
        Single = single word alignment swaps
        Multi = multiple word alignment swaps---only partially worked, as I recall
        Writeout = Shall we write to a file?
        Ten fold = split data into 10 folds for cross eval---NOT IMPLEMENTED (yet)
        Score = the 100,000 scoring metrics we tried
        Device = cuda device (0, 1, or higher) or cpu (-1)
    '''

    # PHRASAL = False
    PHRASAL = True
    # SINGLE = False
    SINGLE = True
    MULTI = False
    # MULTI = True
    WRITEOUT = True
    # WRITEOUT = False
    TENFOLD = False
    # TENFOLD = True
    SCORE = True
    # SCORE = True
    DEVICE = 0
    # DEVICE = 1
    # DEVICE = -1
    batch_size = 1

    ### Ready

    if WRITEOUT:
        output_file = sys.argv[3]

    paraphrases = []

    # TODO there's a better way to do this---naw
    header = [
        'swappable',
        'swap',
        'src',
        'align',
        'para',
        'orig',
        'label',
        'response',
        'cs guess',
        'cs correct',
        'color code 1',
        'color code 2'
    ]

    paraphrases.append(header)

    c = Combine()

    aS = alignSub()

    p = PhraseTable()

    # HACKY PROTOTYPING
    sents = open(sys.argv[1], 'r').readlines()
    labels = aS.get_labels(sents)
    # TODO CHANGE TO LOWER QUINTILE?
    low_freq = aS.get_low_freq(sents)

    if PHRASAL:
        # Attempt at using Sarah's code
        ps = PatternSwap()
        gold_lines = [x.split('\t') for x in open(sys.argv[2], 'r').readlines()]
        patterns = ps.extract_pattern(gold_lines)
        print("patterns", len(patterns))

        checked_patterns = ps.template_check(patterns)
        print("checked_patterns", len(checked_patterns))

        test_num = 0
        # diffed_matches = ps.get_diff(checked_patterns, sents, test_num, until=100)
        # diffed_matches = ps.get_diff(checked_patterns, sents, test_num)
        # TODO FIX THIS HACK
        hacky_sents = ['\t'.join(x) for x in low_freq]
        diffed_matches = ps.get_diff(checked_patterns, hacky_sents, test_num)
        print("diffed_matches", len(diffed_matches))
        # best_matches = ps.refine_matches(diffed_matches)
        # print("best_matches", len(best_matches))

        # phrasal_paraphrases = ps.gen_para(best_matches)
        phrasal_paraphrases = ps.gen_para(diffed_matches)

        # phrasal_paraphrases.insert(0, header)

        paraphrases += phrasal_paraphrases

        print("phrasal_paraphrases", len(phrasal_paraphrases))

    ###############################
    # OLDER STUFF---STILL NEEDED? #
    # YES KEEP FOR GOLDS          #
    # ... AND POSSIBLY NEW ALIGNS #
    ###############################

    golds = []
    # for files in sys.argv[2:]:
    golds = c.read_gold(golds, sys.argv[2])

    # SINGLES
    if SINGLE:
        gold_singles = {}
        gold_singles = aS.get_align(golds, gold_singles)
        gold_sg_para = aS.swap(low_freq, gold_singles)
        paraphrases += gold_sg_para

    # PHRASES
    if MULTI:
        gold_phrases = {}
        gold_phrases = aS.get_range_align(golds, gold_phrases)# , use_phrase = True)
        gold_ph_para = aS.swap(low_freq, gold_phrases)
        paraphrases += gold_ph_para

    if WRITEOUT and not SCORE: aS.writeout(output_file, paraphrases)

    if SCORE:
        para_header = paraphrases.pop(0)
        datas = pd.DataFrame(paraphrases, columns=para_header)

        srcs = datas['src'].values.tolist()
        aligns = datas['align'].values.tolist()
        origs = datas['orig'].values.tolist()
        paras = datas['para'].values.tolist()

        lm = lang_mod()
        sc = Score()

        # Here just for reference
        # header = [
        #     'swappable',
        #     'swap',
        #     'src',
        #     'align',
        #     'para',
        #     'orig',
        #     'label',
        #     'response',
        #     'cs guess',
        #     'cs correct',
        #     'color code 1',
        #     'color code 2'
        # ]

        print("""
                hacky arguments:
                1 = w2v binary file
                2 = glove text file in w2v format
                3 = tsv to score
                4 = original corrected.tsv (2016 VP data)
                5 = Gigaword 5-gram model
                    """)
        # TODO make these back into arguments
        w2v_file = '../data/GoogleNews-vectors-negative300.bin'
        glove_file = '../data/glove.6B.300d.txt.word2vec'
        kenlm_file = '../data/gigaword4.5g.kenlm.bin'
        # temp test

        corrected = sents
        dialog_turn_nums = sc.rebuild_dialogs(corrected)

        ### W2V ###
        print("loading W2V vectors")
        # currently commented out for processing time
        w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_file, binary=True)
        # w2v = gensim.models.KeyedVectors.load_word2vec_format('../data/vectors.300.bin', binary=True)
        # w2v = ''

        ### GloVe ###
        print("loading GloVe vectors")
        # currently commented out for processing time
        glove = gensim.models.KeyedVectors.load_word2vec_format(glove_file, binary=False)

        ### BERT INIT ###
        b = BertBatch('bert-base-uncased', device=DEVICE)

        # pre-b2 = BertBatch('bert-base-uncased', device=0)
        # pre-b4 = BertBatch('bert-base-uncased', device=0)
        # b = BertBatch('bert-large-uncased', device=0)
        # b = BertBatch('/fs/project/white.1240/king/pytorch-pretrained-BERT/examples/newannots_output-34/', device=1)
        # b = BertBatch('/fs/project/white.1240/king/pytorch-pretrained-BERT/examples/newannots_output/', device=1)
        # b = BertBatch('../../pytorch-pretrained-BERT/examples/lm_finetuning/2016-bert-large-uncased-3epochs/', device=0)
        # b = BertBatch('../../pytorch-pretrained-BERT/examples/lm_finetuning/2016-bert-large-uncased-10epochs/', device=0)
        # b = BertBatch('../../pytorch-pretrained-BERT/examples/lm_finetuning/2016-bert-large-uncased-25epochs/', device=0)
        # b = BertBatch('../../pytorch-pretrained-BERT/examples/lm_finetuning/2016-bert-base-uncased-3epochs/', device=0)
        # b = BertBatch('../../pytorch-pretrained-BERT/examples/lm_finetuning/2016-bert-base-uncased-10epochs/', device=0)
        # b = BertBatch('/fs/project/white.1240/king/pytorch-pretrained-BERT/examples/lm_finetuning/2016-bert-base-uncased-25epochs/', device=0)
        # b = BertBatch('/fs/project/white.1240/king/pytorch-pretrained-BERT/examples/lm_finetuning/2016-bert-large-uncased-10epochs/', device=0)

        # KENLM
        print('loading kenlm gigaword 5gram model')
        lm = lang_mod()
        lm.load_lm(kenlm_file)

        def get_bert(srcs, aligns, origs, paras, queue):
            print("Warming up BERT")
            _ = b.extract(srcs, batch_size)
            # b.extract(srcs, batch_size)
            print("Extracting BERT rep for srcs")
            bert_src = b.extract(srcs, batch_size)
            # bert_src_file = pickle.dump(bert_src, open('bert_src.pkl', 'wb'))
            # bert_src = None
            print("Extracting BERT rep for aligns")
            bert_align = b.extract(aligns, batch_size)
            # bert_align_file = pickle.dump(bert_align, open('bert_align.pkl', 'wb'))
            # bert_align = None
            print("Extracting BERT rep for origs")
            bert_orig = b.extract(origs, batch_size)
            # bert_orig_file = pickle.dump(bert_orig, open('bert_orig.pkl', 'wb'))
            # bert_orig = None
            print("Extracting BERT rep for paras")
            bert_para = b.extract(paras, batch_size)
            # bert_para_file = pickle.dump(bert_para, open('bert_para.pkl', 'wb'))
            # bert_para = None
            return ['bert', bert_src, bert_align, bert_orig, bert_para]
            # queue.put(('bert', bert_src, bert_align, bert_orig, bert_para))


        # MULTI-GPU CODE

        # results = Queue()

        # elmo_id = Thread(target=get_elmo, args=(srcs, aligns, origs, paras, results))
        # bert_id = Thread(target=get_bert, args=(srcs, aligns, origs, paras, results))

        # elmo_id.start()
        # bert_id.start()

        # elmo_id.join()
        # bert_id.join()

        # SINGLE GPU CODE

        results = []
        # results.append(get_elmo(srcs, aligns, origs, paras, results))
        results.append(get_bert(srcs, aligns, origs, paras, results))

        # pdb.set_trace()
        # TODO what on earth did I mean with this printout?
        print("eho")
        # pdb.set_trace()
        # elmos, berts = ray.get([elmo_id, bert_id])

        # while not results.empty():
        #     res = results.get()
        for res in results:
            if res[0] == 'elmo':
                elmo_src = res[1]
                elmo_align = res[2]
                elmo_orig = res[3]
                elmo_para = res[4]
            elif res[0] == 'bert':
                bert_src = res[1]
                bert_align = res[2]
                bert_orig = res[3]
                bert_para = res[4]
            else:
                print("error")
                pdb.set_trace()

        print("should be good to go")

        # If you have a GPU, put everything on cuda
        # tokens_tensor = tokens_tensor.to('cuda')
        # segments_tensors = segments_tensors.to('cuda')
        # pdb.set_trace()

        # TODO make this an option or get logging to a different file
        if WRITEOUT:

            header = para_header
            # swap_txt[0].strip().split('\t')

            header = ['dialog',
                      'turn',
                      'glove_src_para_sim',
                      'glove_src_para_dist',
                      'glove_src_para_david',
                      'glove_src_orig_sim',
                      'glove_src_orig_dist',
                      'glove_src_orig_david',
                      'glove_src_align_sim',
                      'glove_src_align_dist',
                      'glove_src_align_david',
                      'glove_orig_para_sim',
                      'glove_orig_para_dist',
                      'glove_orig_para_david',
                      'glove_align_para_sim',
                      'glove_align_para_dist',
                      'glove_align_para_david',
                      'glove_align_orig_sim',
                      'glove_align_orig_dist',
                      'glove_align_orig_david',
                      'w2v_src_para_sim',
                      'w2v_src_para_dist',
                      'w2v_src_para_david',
                      'w2v_src_orig_sim',
                      'w2v_src_orig_dist',
                      'w2v_src_orig_david',
                      'w2v_src_align_sim',
                      'w2v_src_align_dist',
                      'w2v_src_align_david',
                      'w2v_orig_para_sim',
                      'w2v_orig_para_dist',
                      'w2v_orig_para_david',
                      'w2v_align_para_sim',
                      'w2v_align_para_dist',
                      'w2v_align_para_david',
                      'w2v_align_orig_sim',
                      'w2v_align_orig_dist',
                      'w2v_align_orig_david',
                      'bert_src_para_sim',
                      'bert_src_para_dist',
                      'bert_src_para_david',
                      'bert_src_orig_sim',
                      'bert_src_orig_dist',
                      'bert_src_orig_david',
                      'bert_src_align_sim',
                      'bert_src_align_dist',
                      'bert_src_align_david',
                      'bert_orig_para_sim',
                      'bert_orig_para_dist',
                      'bert_orig_para_david',
                      'bert_align_para_sim',
                      'bert_align_para_dist',
                      'bert_align_para_david',
                      'bert_align_orig_sim',
                      'bert_align_orig_dist',
                      'bert_align_orig_david',
                      'bert_sims',
                      'bert_dist',
                      'bert_david',
                      'ng_src_para',
                      'ng_src_orig',
                      'ng_src_align',
                      'ng_orig_para',
                      'ng_align_para',
                      'ng_align_orig',
                      'ng_sum'] + header

            # print('\t'.join(header))
            outfile = open(output_file, 'w')
            outfile.write('\t'.join(header) + '\n')

            line_nmr = 0
            missing = 0
            total = 0

            lost = []

            SANITY = True

            # TODO add sanity check to make sure lengths are all correct
            for line in tqdm(paraphrases):
                # swap_txt[1:]:
                # print("number", total)
                total += 1
                sims = sc.score(line, w2v, glove, lm,
                               bert_src[line_nmr],
                               bert_align[line_nmr],
                               bert_orig[line_nmr],
                               bert_para[line_nmr])
                # print('sims', sims)
                # print('line', line)
                # print('\t'.join(list([str(x) for x in sims]) + line.strip().split('\t')))
                # TODO remove these hacks
                split_line = line
                # split_line = line.strip().split('\t')
                original = split_line[5]
                # print("original", original)
                if len(split_line) == 13:
                    dia_turn = split_line[-1]
                    dia_turn = eval(dia_turn)
                    dial_num = dia_turn[0]
                    turn_num = dia_turn[1]
                    outfile.write('\t'.join(
                        [str(dial_num), str(turn_num)] + list([str(x) for x in sims]) + line) + '\n')
                elif original in dialog_turn_nums:
                    for nums in dialog_turn_nums[original]:
                        dial_num = nums[0]
                        turn_num = nums[1]
                        outfile.write('\t'.join(
                            [str(dial_num), str(turn_num)] + list([str(x) for x in sims]) + [str(x) for x in line]
                        ) + '\n')
                # Sarah's originals are strings that were once arrays
                # TODO I don't think these have to be converted, but
                # double check
                elif '[' in original and ' '.join(eval(original)) in dialog_turn_nums:
                    original = ' '.join(eval(original))
                    for nums in dialog_turn_nums[original]:
                        dial_num = nums[0]
                        turn_num = nums[1]
                        outfile.write('\t'.join(
                            [str(dial_num), str(turn_num)] + list([str(x) for x in sims]) + line) + '\n')
                else:
                    try:
                        outfile.write(
                            '\t'.join([str(999), str(999)] + list([str(x) for x in sims]) + line) + '\n')
                    except:
                        pdb.set_trace()
                    lost.append('\t'.join(list([str(x) for x in sims]) + line) + '\n')
                    missing += 1
                line_nmr += 1

            print("missing", missing, "of", total)


