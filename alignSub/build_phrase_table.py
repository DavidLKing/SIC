import sys
import pdb

class PhraseTable:
    def __init__(self):
        pass

    def str2idx(self, idx_array):
        int_indexes = []
        for nums in sorted(set(idx_array.split(' '))):
            nums = nums.split('-')
            nums = [int(x) for x in nums]
            int_indexes.append(nums)
        return int_indexes

    def conv2range(self, indexes):
        """
        simple algorithm to get ranges
        for i in first number:
            create [i, ]
            append everything related to i
        in new array, for everything range:
            create [ , j], where j is a range and every first number with the same range fors in the first place
        :param indexes:
        :return:
        """
        tgt_ranges = {}
        for num in indexes:
            if num[0] not in tgt_ranges:
                tgt_ranges[num[0]] = []
            tgt_ranges[num[0]].append(num[1])
        dict_ranges = {}
        for num in tgt_ranges:
            t_range = ' '.join([str(x) for x in tgt_ranges[num]])
            if t_range not in dict_ranges:
                dict_ranges[t_range] = []
            dict_ranges[t_range].append(num)
        full_ranges = []
        for t_range in dict_ranges:
            s_range = dict_ranges[t_range]
            t_range = [int(x) for x in t_range.split(' ')]
            full_ranges.append([s_range, t_range])
        return full_ranges

    def gen_phrase(self, sent, indexes):
        # print(indexes)
        prev = None
        phrase = ''
        if type(indexes) == int:
            # TODO weird even handling.
            indexes = [indexes]
        try:
            for num in indexes:
                if prev == None or num - prev == 1:
                    phrase += sent[num]
                    phrase += ' '
                else:
                    phrase += '_ '
                prev = num
        except:
            pdb.set_trace()
        return phrase.strip()

    def update_phrase_table(self, src, tgt, src_phrase, tgt_phrase, phrase_table):
        '''
        Data structure looks like:
        {
            src_1 :
                {
                    tgt_1 :
                        src_sents = [s1, s2, s3, ...],
                        tgt_sents = [t1, t2, t3, ...]
                    tgt_2 :
                        ...
                }
            src_2 :
                ...
        }
        '''
        if src_phrase != tgt_phrase:
            if src_phrase not in phrase_table:
                phrase_table[src_phrase] = {}
            if tgt_phrase not in phrase_table[src_phrase]:
                phrase_table[src_phrase][tgt_phrase] = {}
                phrase_table[src_phrase][tgt_phrase]['src'] = []
                phrase_table[src_phrase][tgt_phrase]['align'] = []
            phrase_table[src_phrase][tgt_phrase]['src'].append(' '.join(src))
            phrase_table[src_phrase][tgt_phrase]['align'].append(' '.join(tgt))
        '''
        other idea:
        swap = (src_phrase, tgt_phrase)
        if src_phrase != tgt_phrase:
            if swap not in phrase_table:
                phrase_table[swap] = {}
                phrase_table[swap]['src_sents'] = []
                phrase_table[swap]['tgt_sents'] = []
            if src not in phrase_table[swap]['src_sents']:
                phrase_table[swap]['src_sents'].append(src)
            if tgt not in phrase_table[swap]['tgt_sents']:
                phrase_table[swap]['tgt_sents'].append(tgt)
        '''
        return phrase_table

    # def align_single(self, src, tgt, indexes, phrase_table):
    #     # mocked up from PhraseTable.py to not create phrases
    #     src = src.split()
    #     tgt = tgt.split()
    #     # indexes = p.conv2range(indexes)
    #     for pair in indexes:
    #         # src_phrase = self.gen_phrase(src, pair[0])
    #         # tgt_phrase = self.gen_phrase(tgt, pair[1])
    #         src_phrase = src[pair[0]]
    #         tgt_phrase = tgt[pair[1]]
    #         if src_phrase != tgt_phrase:
    #             if src_phrase not in phrase_table:
    #                 phrase_table[src_phrase] = []
    #             if tgt_phrase not in phrase_table[src_phrase]:
    #                 phrase_table[src_phrase].append(tgt_phrase)
    #     return phrase_table

    def align(self, src, tgt, indexes, phrase_table):
        src = src.split()
        tgt = tgt.split()
        # pdb.set_trace()
        for pair in indexes:
            src_phrase = self.gen_phrase(src, pair[0])
            tgt_phrase = self.gen_phrase(tgt, pair[1])
            phrase_table = self.update_phrase_table(src, tgt, src_phrase, tgt_phrase, phrase_table)
            # Reverse it, too
            phrase_table = self.update_phrase_table(tgt, src, tgt_phrase, src_phrase, phrase_table)
        return phrase_table

    def build(self, src_tgt_array):
        # I'm not tied to this being a dict
        phrase_table = {}
        for pair in src_tgt_array:
            src = pair[0]
            tgt = pair[1]
            indexes = pair[2]
            # TODO fix this redundency in the combineGold read_gold function
            if indexes != '':
                # print(src)
                # print(tgt)
                # print(indexes)
                indexes = self.str2idx(indexes)
                # pdb.set_trace()
                phrase_table = self.align(src, tgt, indexes, phrase_table)
            else:
                print("Error! There's a bug in this entry. alignments cannot be empty.",
                      "\nPlease see this line. This is currently an error.", pair)
        return phrase_table


