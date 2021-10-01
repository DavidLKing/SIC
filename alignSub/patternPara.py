import pickle
import csv
import difflib
import nltk
import pdb
from tqdm import tqdm

class PatternSwap:
    def __init__(self):
        pass

    def match(self, line):
        pdb.set_trace()

    def apply(self, line):
        pass

    def extract_pattern(self, lines):
        '''
        From makeAutTemplates4.py
        :param lines:
        :return:
        '''

        stopwords = {'is', 'are', 'what', 'who', 'where',
                     'when', 'how', 'why', 'you', 'your',
                     'have', 'has', 'be', 'to', 'do'}

        templates = []

        for i, line in enumerate(lines):
            if line[6] == '1':
                aligns, aligns2, finalaligns = [], [], []
                sent1a = line[1].split()
                sent1b = nltk.pos_tag(sent1a)
                sent2a = line[3].split()
                sent2b = nltk.pos_tag(sent2a)
                sure = line[7].split()
                # try:
                poss = line[8].split()
                # except:
                # TODO this isn't working, why did Sarah leave it here
                # pdb.set_trace()
                align = sure + poss
                variable = 0
                for item in align:
                    x, y = item.split('-')
                    aligns.append(([int(x)], [int(y)]))
                for pair in align:
                    x, y = pair.split('-')
                    for item in aligns:
                        if int(x) in item[0]:
                            if not int(y) in item[1]:
                                item[1].append(int(y))
                        elif int(y) in item[1]:
                            item[0].append(int(x))
                for item in aligns:
                    left = sorted(item[0])
                    right = sorted(item[1])
                    if not (left, right) in aligns2:
                        aligns2.append((left, right))
                for (left, right) in aligns2:
                    try:
                        l, r = 0, 0
                        for i in left:
                            if not sent1a[i] in stopwords: l += 1
                        for i in right:
                            if not sent2a[i] in stopwords: r += 1
                        if l > 0 or r > 0:
                            finalaligns.append((left, right))
                    except:
                        # TODO why is sara's code breaking again?!
                        continue
                ## no need to change anything so far
                for i in range(0, len(finalaligns)):
                    tpl = finalaligns[i]
                    left, right = tpl[0], tpl[1]
                    if len(left) == 1 and len(right) == 1:
                        ## ADD POS TAG
                        sent1a[left[0]] = "$" + str(variable) + '/' + sent1b[left[0]][1]
                        sent2a[right[0]] = "$" + str(variable) + '/' + sent2b[right[0]][1]
                        variable += 1
                    else:
                        j, k = len(left) - 1, len(right) - 1
                        tag_left = []
                        tag_right = []
                        while j >= 0:
                            if j == 0:
                                tag_left.insert(0, sent1b[left[j]][1])
                                sent1a[left[j]] = "$" + str(variable) + '//' + str(tag_left)
                                j -= 1
                            else:
                                tag_left.insert(0, sent1b[left[j]][1])
                                sent1a[left[j]] = "REPLACE"
                                j -= 1
                        while k >= 0:
                            if k == 0:
                                tag_right.insert(0, sent2b[right[k]][1])
                                sent2a[right[k]] = "$" + str(variable) + '//' + str(tag_right)
                                k -= 1
                            else:
                                sent2a[right[k]] = "REPLACE"
                                tag_right.insert(0, sent2b[right[k]][1])
                                k -= 1
                        variable += 1
                while "REPLACE" in sent1a: sent1a.remove("REPLACE")
                while "REPLACE" in sent2a: sent2a.remove("REPLACE")
                i, j = 0, 0
                for word in sent1a:
                    if "$" not in word:
                        i += 1
                if i != 0 and i != len(sent1a):
                    for word in sent2a:
                        if "$" not in word:
                            j += 1
                    if j != 0 and j != len(sent2a):
                        if not sent1a == sent2a:
                            templates.append((sent1a, sent2a, line[1], line[3]))
                            templates.append((sent2a, sent1a, line[3], line[1]))
        return templates

    def third_extract_pattern(self, lines):
        stopwords = {'is', 'are', 'what', 'who',
                     'where', 'when', 'how', 'why',
                     'you', 'your', 'have', 'has', 'be',
                     'to', 'them'}
        templates = []
        for i, line in enumerate(lines):
            if line[6] == '1':
                aligns, aligns2, finalaligns = [], [], []
                sent1a = line[1].split()
                sent1b = nltk.pos_tag(sent1a)
                sent2a = line[3].split()
                sent2b = nltk.pos_tag(sent2a)
                sure = line[7].split()
                poss = line[8].split()
                align = sure + poss
                variable = 0
                for item in align:
                    x, y = item.split('-')
                    aligns.append(([int(x)], [int(y)]))
                for pair in align:
                    x, y = pair.split('-')
                    for item in aligns:
                        if int(x) in item[0]:
                            if not int(y) in item[1]:
                                item[1].append(int(y))
                        elif int(y) in item[1]:
                            item[0].append(int(x))
                for item in aligns:
                    left = sorted(item[0])
                    right = sorted(item[1])
                    if not (left, right) in aligns2:
                        aligns2.append((left, right))
                for (left, right) in aligns2:
                    l, r = 0, 0
                    for i in left:
                        if not sent1a[i] in stopwords: l += 1
                    for i in right:
                        if not sent2a[i] in stopwords: r += 1
                    if l > 0 or r > 0:
                        finalaligns.append((left, right))
                for i in range(0, len(finalaligns)):
                    tpl = finalaligns[i]
                    left, right = tpl[0], tpl[1]
                    if len(left) == 1 and len(right) == 1:
                        ## ADD POS TAG
                        sent1a[left[0]] = "$" + str(variable) + '/' + sent1b[left[0]][1]
                        sent2a[right[0]] = "$" + str(variable) + '/' + sent2b[right[0]][1]
                        variable += 1
                    else:
                        j, k = 0, 0
                        while j < len(left):
                            if j == 0:
                                sent1a[left[j]] = "$" + str(variable) + '/' + str(len(left))
                                j += 1
                            else:
                                sent1a[left[j]] = "REPLACE"
                                j += 1
                        while k < len(right):
                            if k == 0:
                                sent2a[right[k]] = "$" + str(variable) + '/' + str(len(right))
                                k += 1
                            else:
                                sent2a[right[k]] = "REPLACE"
                                k += 1
                        variable += 1
                while "REPLACE" in sent1a: sent1a.remove("REPLACE")
                while "REPLACE" in sent2a: sent2a.remove("REPLACE")
                i, j = 0, 0
                for word in sent1a:
                    if not "$" in word:
                        i += 1
                if i != 0 and i != len(sent1a):
                    for word in sent2a:
                        if not "$" in word:
                            j += 1
                    if j != 0 and j != len(sent2a):
                        if not sent1a == sent2a:
                            templates.append((sent1a, sent2a))
                            templates.append((sent2a, sent1a))
        return templates

    def first_extract_pattern(self, lines):
        stopwords = {'is', 'are', 'what', 'who', 'where', 'when', 'how', 'why', 'you', 'your', 'have', 'has', 'be', 'to', 'them'}
        templates = []
        for i, line in enumerate(lines):
            if line[6] == '1':
                aligns, aligns2, finalaligns = [],[],[]
                sent1 = line[1].split()
                sent1 = nltk.pos_tag(sent1)
                sent2 = line[3].split()
                sent2 = nltk.pos_tag(sent2)
                sure = line[7].split()
                poss = line[8].split()
                align = sure+poss
                for item in align:
                    x,y = item.split('-')
                    aligns.append(([int(x)],[int(y)]))
                for pair in align:
                    x,y = pair.split('-')
                    for item in aligns:
                        if int(x) in item[0]:
                            if not int(y) in item[1]:
                                item[1].append(int(y))
                        elif int(y) in item[1]:
                            item[0].append(int(x))
                for item in aligns:
                    left = sorted(item[0])
                    right = sorted(item[1])
                    if not (left, right) in aligns2:
                        aligns2.append((left, right))
                for (left,right) in aligns2:
                    l,r = 0,0
                    for i in left:
                        # l += 1
                        if not sent1[i] in stopwords: l+=1
                    for i in right:
                        # r += 1
                        if not sent2[i] in stopwords: r+=1
                    if l > 0 or r > 0:
                        finalaligns.append((left,right))
                for i in range(0, len(finalaligns)):
                    tpl = finalaligns[i]
                    left,right = tpl[0],tpl[1]
                    j,k = 0,0
                    while j < len(left):
                        if j==0:
                            sent1[left[j]] = "$"+str(i)
                            j+=1
                        else:
                            sent1[left[j]] = "REPLACE"
                            j+=1
                    while k < len(right):
                        if k==0:
                            sent2[right[k]] = "$"+str(i)
                            k+=1
                        else:
                            sent2[right[k]] = "REPLACE"
                            k+=1
                while "REPLACE" in sent1: sent1.remove("REPLACE")
                while "REPLACE" in sent2: sent2.remove("REPLACE")
                i,j = 0,0
                for word in sent1:
                    if not "$" in word:
                        i+=1
                if i != 0 and i != len(sent1):
                    for word in sent2:
                        if not "$" in word:
                            j += 1
                    if j != 0 and j != len(sent2):
                        if not sent1 == sent2:
                            templates.append((sent1, sent2, line[1], line[3]))
        return templates

    def get_diff(self, checked_patterns, sents, test_num, until=None):
        matches = []
        labels = []
        # testing
        for (pattern_1, pattern_2, sent_1, sent_2) in tqdm(checked_patterns):
            # range for testing
            if until:
                end = until
            else:
                end = len(checked_patterns) + 1
            # for (pattern_1, pattern_2, sent_1, sent_2) in tqdm(checked_patterns[0:until]):
            for line in tqdm(sents[0:end]):
                if not line.startswith('#'):
                    line = line.split('\t')
                    label = line[3]
                    sentence = line[0].split()
                    test_1 = self.doDiff(label, pattern_2, pattern_1, sentence, sent_1, sent_2, test_num)
                    test_num += 1
                    if test_1:
                        matches.append(test_1)
        return matches


    def doDiff(self, lab, trans, sent, wilks, source1, source2, dialTurn):
        diff = difflib.ndiff(sent, wilks)
        pos, neg, w, c, tag = 0, 0, 0, 0, 0
        for item in sent:
            if '$' not in item:
                w += 1
        dif = []
        for item in diff:
            dif.append(item)
        for i in range(0, len(dif) + 1):
            if i == len(dif):
                if (neg > tag and c == w and neg <= pos) or (neg == tag and neg == pos and c == w):
                    return (lab, wilks, sent, trans, source1, source2, dialTurn)
            elif dif[i][0] == ' ':
                c += 1
                if neg == pos and neg == tag:
                    neg, pos, tag = 0, 0, 0
                elif neg <= pos and neg > tag:
                    neg, pos, tag = 0, 0, 0
                else:
                    break
            elif dif[i][0] == '-':
                neg += 1
                if '/' in dif[i][2:]:
                    tag += 1
            elif dif[i][0] == '+':
                if len(dif[i][2:].split('//')) > 1:
                    pos += len(list(dif[i][2:].split('//')[1]))
                else:
                    pos += 1

    def template_check(self, patterns):
        '''
        based on templateCheck4.py
        :param patterns:
        :return:
        '''
        used = []
        fails = 0
        for (sent1, sent2, source1, source2) in patterns:
            left = 0
            right = 0
            for word in sent1:
                if '$' in word:
                    left += 1
            for word in sent2:
                if '$' in word:
                    right += 1
            if left != right:
                break
            inds1 = []
            success = 0
            for word in sent1:
                if '$' in word:
                    word = word.split('/')
                    inds1.append(int(word[0][1:]))
            if len(inds1) > 2:
                for i in range(1, len(inds1) - 1):
                    if inds1[i] == inds1[i - 1] + 1 and inds1[i + 1] == inds1[i] + 1:
                        success += 1
                    if success == len(inds1) - 2:
                        if not (sent1, sent2, source1, source2) in used:
                            used.append((sent1, sent2, source1, source2))
                        else:
                            fails += 1
                    else:
                        self.alter_variables(sent1, sent2, fails, used, source1, source2)
            elif inds1 == [0] or inds1 == [0, 1]:
                if not (sent1, sent2, source1, source2) in used:
                    used.append((sent1, sent2, source1, source2))
                else:
                    fails += 1
            else:
                self.alter_variables(sent1, sent2, fails, used, source1, source2)

        for (sent1, sent2, source1, source2) in patterns:
            if not (sent1, sent2, source1, source2) in used:
                i = 0
                f = False
                for word in sent1:
                    if '$' in word:
                        if int(word[1]) == i:
                            i += 1
                        else:
                            f = True
                if f is False:
                    if not (sent1, sent2) in used:
                        used.append((sent1, sent2, source1, source2))
        return used

    def alter_variables(self, sent1, sent2, failed, used, source1, source2):
        for word in sent1:
            if '$' in word:
                place = sent1.index(word)
                splitWord = word.split('//')
                if len(splitWord) == 1:
                    splitWord = word.split('/')
                ind = splitWord[0][1:]
                if 'A' not in ind:
                    new = word[0] + 'A' + ind + word[2:]
                    sent1.remove(word)
                    sent1.insert(place, new)
        for word2 in sent2:
            if '$' in word2:
                place = sent2.index(word2)
                splitWord = word2.split('//')
                if len(splitWord) == 1:
                    splitWord = word2.split('/')
                ind = splitWord[0][1:]
                if 'A' not in ind:
                    new = word2[0] + 'A' + ind + word2[2:]
                    sent2.remove(word2)
                    sent2.insert(place, new)
        var = 0
        for f in range(0, len(sent1)):
            if '$' in sent1[f]:
                double = True
                word = sent1[f].split('//')
                if len(word) == 1:
                    word = sent1[f].split('/')
                    double = False
                ind = word[0][1:]
                if double is True:
                    sent1[f] = word[0][0] + str(var) + '//' + word[1]
                else:
                    sent1[f] = word[0][0] + str(var) + '/' + word[1]
                for l in range(0, len(sent2)):
                    if '$' in sent2[l]:
                        word2 = sent2[l].split('//')
                        if len(word2) == 1:
                            word2 = sent2[l].split('/')
                        if word2[0][1:] == ind:
                            if double == True:
                                sent2[l] = word2[0][0] + str(var) + '//' + word2[1]
                            else:
                                sent2[l] = word2[0][0] + str(var) + '/' + word2[1]

                var += 1
        fail = 0
        for word in sent2:
            if '$' in word:
                ind = word.split('/')[0][1:]
                try:
                    if int(ind) >= var:
                        fail += 0
                except ValueError:
                    print('sent1', sent1)
                    print('sent2', sent2)
        if not (sent1, sent2, source1, source2) in used and fail == 0:
            used.append((sent1, sent2, source1, source2))
        else:
            failed += 1

    def refine_matches(self, matches):
        '''
        based on Sarah's refineMatches4
        :param patterns:
        :return:
        '''
        # this first part removes matches where two words are adjacent in the template
        # but not in the Wilkins sentence
        # label is wilkins.csv line[3] "gold label"
        # wilkins is wilkins.csv line[0] or line[1] "observed sentence in dialogue"
        # matched is the left side of the template which matched on the Wilkins sentence
        # trans is the right half of that template
        # the rest are just being pulled through

        bettermatches = []

        for (label, wilkins, matched, trans, source1, source2, dialTurn) in matches:
            for i in range(1, len(matched) + 1):
                if not i == len(matched):
                    if '$' not in matched[i] and '$' not in matched[i - 1]:
                        ind = wilkins.index(matched[i - 1])
                        if not wilkins[ind:].index(matched[i]) == 1:
                            break
                else:
                    bettermatches.append((label, wilkins, matched, trans, source1, source2, dialTurn))

        # variables are the same as above
        # this section makes sure each variable is filled with the correct number of words
        # with the correct POS tags

        bestmatches = []

        def check_match(wilkins, matched):
            wilkstag = nltk.pos_tag(wilkins)
            ind = 0
            for i in range(0, len(matched) + 1):
                if i != len(matched):
                    if '$' in matched[i] and '//' in matched[i]:
                        ind += len(list(matched[i].split('//')[1]))
                        pass
                        # pos_list = list(matched[i].split('//')[1])      # SECTION 5
                        # for l in range(0, len(pos_list)):
                        #     if not pos_list[l] == wilkstag[ind][1]:
                        #         #fail1 += 1
                        #         return False
                        # else: fail1 += 1
                        #     ind += 1
                    elif '$' in matched[i] and '/' in matched[i]:
                        tag = matched[i].split('/')[1]
                        try:
                            ttag = wilkstag[ind][1]
                        except IndexError:
                            return False
                        if not tag == ttag:
                            # fail2 += 1
                            return False
                        # else: fail2 += 1
                        ind += 1
                    else:
                        try:
                            word = wilkstag[ind][0]
                        except IndexError:
                            return False
                        if not matched[i] == word:
                            # fail3 += 1
                            return False
                        # else: fail3 += 1
                        ind += 1
                else:
                    return True

        for (label, wilkins, matched, trans, source1, source2, dialTurn) in bettermatches:
            if check_match(wilkins, matched) is True:
                bestmatches.append((label, wilkins, matched, trans, source1, source2, dialTurn))

        for (label, wilkins, matched, trans, source1, source2, dialTurn) in bettermatches:
            if not (label, wilkins, matched, trans, source1, source2, dialTurn) in bestmatches:
                wilkstag = nltk.pos_tag(wilkins)
                match = 0
                for item in matched:
                    if len(item.split('//')) > 1:
                        match += len(list(item.split('//')[1]))
                    else:
                        match += 1
                if len(matched[0].split('//')) > 1:
                    tag = list(matched[0].split('//')[1])[0]
                    for l in range(0, len(wilkstag)):
                        if wilkstag[l][1] == tag:
                            ind = l
                            break
                    if not ind + match > len(wilkstag):
                        if check_match(wilkins, matched) is True:
                            bestmatches.append((label, wilkins, matched, trans, source1, source2, dialTurn))
                elif len(matched[0].split('/')) > 1:
                    tag = matched[0].split('/')[1]
                    for l in range(0, len(wilkstag)):
                        if wilkstag[l][1] == tag:
                            ind = l
                            break
                    if not ind + match > len(wilkstag):
                        if check_match(wilkins, matched) is True:
                            bestmatches.append((label, wilkins, matched, trans, source1, source2, dialTurn))
                else:
                    for l in range(0, len(wilkstag)):
                        if wilkstag[l][0] == matched[0]:
                            ind == l
                            break
                    if not ind + match > len(wilkstag):
                        if check_match(wilkins, matched) is True:
                            bestmatches.append((label, wilkins, matched, trans, source1, source2, dialTurn))

        return bestmatches

    def extract_variables(self, par, matched, wilks, trans, j, fail):
        # feed in the matched template, and everything after 'beg' in wilkins
        for i in range(0, len(matched)):
            if len(matched[i].split('//')) > 1:
                var = []
                num = len(list(matched[i].split('//')))
                for l in range(j, num):
                    var.append(wilks[l])
                var = ' '.join(var)
                j += num
                par.append(var)
            elif len(matched[i].split('/')) > 1:
                try:
                    var = wilks[j]
                except IndexError:
                    print('wilks', wilks)
                    print('j', j)
                    print('matched', matched)
                    par = None
                    fail += 1
                    return par, j, fail
                j += 1
                par.append(var)
            else:
                j += 1
        return par, j, fail

    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return 'a'
        elif treebank_tag.startswith('V'):
            return 'a'
        elif treebank_tag.startswith('N'):
            return 'n'
        elif treebank_tag.startswith('R'):
            return 'r'
        else:
            return None

    def change_pos(self, word, pos1, pos2):
        # DLK HACK
        lookup = {}
        # / HACK
        wnl = nltk.stem.WordNetLemmatizer()
        if pos1 != pos2:
            pos = self.get_wordnet_pos(pos1)
            if pos is None:
                lemma = wnl.lemmatize(word)
            else:
                lemma = wnl.lemmatize(word, pos)
            try:
                new = lookup[lemma][pos2]
            except KeyError:
                new = word
        else:
            return word
        return new

    def look_up(self, variable, trans_list, match_list):
        '''
        I think this isn't doing anything since we don't have "lookup2"
        :param variable:
        :param trans_list:
        :param match_list:
        :return:
        '''
        # DLK HACK
        lookup2 = {}
        # / HACK
        if trans_list == match_list:
            return variable
        else:
            try:
                var = lookup2[variable][len(trans_list)]
                for item in var:
                    return item
            except KeyError:
                return variable

    def make_par(self, trans, paraphrase, par, i, sent):
        for l in range(0, len(trans)):
            if '$' not in trans[l]:
                paraphrase.append(trans[l])
            elif len(trans[l].split('//')) > 1:
                var = trans[l].split('//')[0]
                trans_list = list(trans[l].split('//')[1])
                for item in sent:
                    if '$' in item and item.split('/')[0] == var:
                        match_list = list(item.split('//')[1])
                paraphrase.append(self.look_up(par[i], trans_list, match_list))
                i += 1
            elif len(trans[l].split('/')) > 1:
                var = trans[l].split('/')[0]
                trans_pos = trans[l].split('/')[1]
                for item in sent:
                    if '$' in item and item.split('/')[0] == var:
                        match_pos = item.split('/')[1]
                paraphrase.append(self.change_pos(par[i], match_pos, trans_pos))
                i += 1
        return paraphrase

    def gen_para(self, matches):
        para_list = []
        fail = 0
        called, changed, breaks, paraphrases, extract = 0, 0, 0, 0, 0
        secCall, check, check2 = 0, 0, 0
        # with open('paraphrasesOnePOSManyLength.tsv', 'w') as write:
        #     writer = csv.writer(write, delimiter='\t')
        #     writer.writerow(['TEMP MATCH (SWAPPABLE)', 'TEMP TRANSFORMATION (SWAP)', 'SOURCE LEFT (SOURCE ALIGN)',
        #                      'SOURCE RIGHT (SOURCE ALIGN)', 'PARAPHRASE', 'ORIGINAL', 'LABEL', 'DIALOGUE, TURN'])
        for (lab, wilks, sent, trans, source1, source2, dialTurn) in matches:
            wilks_string = ' '.join(wilks)
            secCall += 1
            j, f, beg, par = 0, 0, [], []
            # if '$' not in sent[0]:
            #    j = wilks.index(sent[0])+1
            #   for i in range(0, wilks.index(sent[0])):
            #      beg.append(wilks[i])
            # beg = ' '.join(beg)
            # if not beg == '':
            #   par.append(beg)
            par, j, fail = self.extract_variables(par, sent, wilks, trans, j, fail)
            if par is not None:
                check += 1
                if j < len(wilks):
                    add = []
                    for i in range(j, len(wilks)):
                        add.append(wilks[i])
                    add = ' '.join(add)
                    if not add == '':
                        par.append(add)

                # par contains the information from wilks stored within each variable
                # below the paraphrase is created from this information
                #             changes = get_changes(par, gold_singles, gold_phrases)
                #            changes, singles, multiples = get_changes(par, elmo_singles, elmo_phrases)
                #            changes = set(changes)
                v, paraphrase, i = 0, [], 0
                for item in trans:
                    if '$' in item:
                        v += 1
                if v == len(par):
                    paraphrase = self.make_par(trans, paraphrase, par, i, sent)
                elif (v + 1) == len(par):
                    check2 += 1
                    if '$' in sent[0]:
                        paraphrase = self.make_par(trans, paraphrase, par, i, sent)
                        paraphrase.append(par[-1])
                    else:
                        paraphrase.append(par[0])
                        i += 1
                        paraphrase = self.make_par(trans, paraphrase, par, i, sent)
                elif (v + 2) == len(par):
                    paraphrase.append(par[0])
                    i = 1
                    paraphrase = self.make_par(trans, paraphrase, par, i, sent)
                    paraphrase.append(par[-1])
                else:
                    print('ERROR')
                    print('V', v)
                    print('par', len(par), par)

                paraphrase = ' '.join(paraphrase)
                paraphrases += 1
                # row = [sent, trans, source1, source2, paraphrase, wilks, lab, dialTurn]
                row = [' '.join(sent), ' '.join(trans), source1, source2, paraphrase, wilks_string, lab, dialTurn]
                para_list.append(row)
                # writer.writerow(row)
        return para_list
