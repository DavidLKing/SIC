# from __future__ import print_function
import difflib
import subprocess
import signal
import sys
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import gensim
import ast
from affixcheck import affixes
# import cPickle as pkl
import pickle as pkl
import gensim
import arpa
import kenlm
import operator
from collections import OrderedDict
import pdb
import math

nltk.data.path.append("/scratch2/king/nltk_data")

class lexsub:
    def __init__(self, outfile):
        self.wnl = WordNetLemmatizer()
        self.diff = affixes()
        self.prec_num = 0
        self.kept = 0
        self.rejected = 0
        self.prec_per_query = []
        self.mean = lambda x: sum(x) / len(x)
        self.stopWords = set(stopwords.words('english'))
        self.outcsv = open('outputs/' + outfile, 'w')
        self.didnt_make_it = open('outputs/' + outfile[:-4] + '-didnt-make-it.tsv', 'w')
        header = '\t'.join(['label', 'orginal sent', 'target word', 'chosen word', 'paraphrased', 'seen in vars', 'diff in score\n'])
        # self.outcsv.write('\t'.join([source, label, original, orig_word, newSent, "\n"]))
        self.outcsv.write(header)
        self.didnt_make_it.write(header)
        self.wordnet = False
        self.w2v = False
        self.sem_clust = False
        self.kept_map = []
        self.reject_map = []
        self.mode = '--normal'
        # self.mode = '--first'
        self.select_mode = '--ngram'
        # sentence = sys.argv[1:]
        # print sentence

    def load_paraphrases(self, para_pkl):
        self.paraphrases = pkl.load(open(para_pkl, 'rb'))
        pdb.set_trace()
        print('Successfully loaded', len(self.paraphrases), 'paraphrase labels')

    def load_w2v(self, vectors):
        try:
            print("Loading word2vec vectors")
            # self.model = gensim.models.Word2Vec.load_word2vec_format(vectors, binary=True)
            self.model = gensim.models.KeyedVectors.load_word2vec_format(vectors, binary=True)
        except:
            pdb.set_trace()
    
    def load_glove(self, vectors):
        print("Loading glove vectors")
        self.glove = gensim.models.KeyedVectors.load_word2vec_format(vectors, binary=False)

    def query(self, word, pos):
        orig_word = word
        # if pos in ['ADJP', 'ADVP', 'IN', 'INTJ', \
        #            'JJ', 'JJR', 'JJS', 'NN', 'NNP', \
        #            'NNPS', 'NNS', 'NP', 'PP', 'RB', \
        #            'VBD', 'VBG', 'VBN', 'VB', 'VBP', \
        #            'VBZ', 'VP'] and \
        # word.lower() not in ['do', 'is', 'are', 'you', 'have']:
        # print("Checking for stopwords:", )
        if word.lower() not in self.stopWords:
            # print('trying', word, pos)
            command = 'zgrep '
            word = '"|||\s' + word + '\s" '
            # command += 'NN '
            # prevent "dr." from matching "dra", "drb", "dry", etc...
            command += word.replace(".", "\.")
            directory = 'ppdb_sense_clusters/ppdb-2.0-xxl-all-clusters-' + pos + '.numbered_max.gz'
            command += directory
            print('\tcommand is ', command)
            # try:
                # worked on python2, but we need python3 for arpa to work
                # output = subprocess.check_output(command, shell=True, preexec_fn=lambda: signal.signal(signal.SIGPIPE, signal.SIG_DFL))
            try:
                output = str(subprocess.check_output(command, shell=True)).replace("\\n'", "").replace("b'", "")
                output = output.split(' ||| ')
            except:
                output = [pos, orig_word, "{'1' : []}"]
                print('error 1 status with:', command)
            # print('output is', output.replace("\\n'", "").replace("b'", ""))
            # print('type of output is ', type(output), len(output))
            # print('type after ast', type(ast))
        else:
            print("\t", orig_word, 'is a stop word')
            output = [pos, orig_word, "{'1' : []}"]
        # print("output", output)
        # print("len(output)", len(output))
        assert(len(output) == 3)
        try:
            cmd_output = ast.literal_eval(output[2])
        except:
            print("Error parsing literal grep output as a python3 dictionary for the following command:\n", command)
            # TODO why is this happening?
            cmd_output = ast.literal_eval([pos, orig_word, "{'1' : []}"][2])
        return output[0], output[1], cmd_output

    def breaksent(self, sentence):
        self.orig = sentence
        tokenized = word_tokenize(sentence)
        print("original sentence:", self.orig)
        print("broken sentence:", tokenized)
        return nltk.pos_tag(tokenized)

    def get_words(self, label):
        words = set()
        # print('label', label)
        for dialog in label:
            paraphrase = dialog[0]
            # print("get word paraphrase", paraphrase)
            for word in paraphrase.split():
                # print('\tword', word)
                if word not in self.stopWords:
                    words.add(word)
        # for word in label.split():
        #     words.add(word)
        # print('words', words)
        return words

    def replace(self, sentArray, index, cluster, words, label):
        new_para = []
        original = ' '.join(sentArray)
        orig_score = self.sent_scorer(sentArray, sentArray[index])
        # orig_score = self.lm.score(original)
        print('\tOriginal', original)
        # seen = self.paraphrases[original]
        # words = self.get_words(cluster)
        # print("Paraphrases!", seen)
        print('Looking at', sentArray[index])
        print('current index is', index)
        # print("sentArray", sentArray)
        # print('\twords in other variants:')
        # for word in words:
        #     print('\t\t', word)
        # print('\t', words)
        # print("\tcluster =", cluster)
        # print('testing:', ' '.join(sentArray))
        # print('\tcluster:', cluster)
        # if cluster != []:
            # for word_w_logprob in cluster:
                # word = word_w_logprob[0]
        for paraphrase_w_logprob in cluster:
            # logprob is unused
            # logprob = paraphrase_w_logprob[1]
            paraphrase = paraphrase_w_logprob[0]
            orig_word = sentArray[index]
            print('\t', orig_word, '-->', paraphrase)
            print('\t in other variant?', paraphrase in words)
            sentArray[index] = paraphrase
            newSent = ' '.join(sentArray)
            print('\t\tNEWSENT',newSent)
            # new_score = self.lm.score(newSent)
            new_score = self.sent_scorer(sentArray, paraphrase)
            # pdb.set_trace()
            change = orig_score - new_score
            # self.outcsv.write('\t'.join([label, original, orig_word, newSent, str(change), "\n"]))
            new_para.append((label, original, orig_word, paraphrase, newSent, paraphrase in words, change))
            # pdb.set_trace()
            sentArray[index] = orig_word
            print("after sentArray")
            for word in sentArray:
                print("\t", word)
        return new_para

    def sent_change(self, orig_score, new_score):
        pass

    def sim_to_logprob(self, cos_sim):
        """
        convert a cosign similarity score to a pseudo logprob
        cos_sim + 1 / 2 --> then take the log
        """
        return math.log((cos_sim + 1) / 2)
    

    def sent_scorer(self, sentence, word):
        """
        score sentence based on self.select_mode
        ngram = self.lm.score(sentence)
        glove = avg of all content words to swapped word

        """
        if self.select_mode == '--ngram':
            return self.lm.score(' '.join(sentence))
        elif self.select_mode == '--ngram-word':
            return self.lm.score(word)
        elif self.select_mode in ['--glove', '--glo5', '--glopos']:
            sims = []
            if word in self.glove:
                for sent_words in sentence:
                    if sent_words not in self.stopWords and sent_words in self.glove:
                        sims.append(self.sim_to_logprob(self.glove.similarity(sent_words, word)))
                    # TODO for OOVs, should we do anything?
                    # elif sent_words not in self.stopWords and sent_word not in self.glove:
                if sims == []:
                    sims = [0.0]
                return self.mean(sims)
            else:
                # When swapped word is an OOV
                return self.sim_to_logprob(0.0)
        elif self.select_mode == '--glabs':
            sims = []
            if word in self.glove:
                for sent_words in sentence:
                    if sent_words not in self.stopWords and sent_words in self.glove:
                        sims.append(self.sim_to_logprob(abs(self.glove.similarity(sent_words, word))))
                    # TODO for OOVs, should we do anything?
                    # elif sent_words not in self.stopWords and sent_word not in self.glove:
                if sims == []:
                    sims = [0.0]
                return self.mean(sims)
            else:
                # When swapped word is an OOV
                return self.sim_to_logprob(0.0)
        elif self.select_mode in ['--combo', '--com5', '--compos']:
            sims = []
            if word in self.glove:
                for sent_words in sentence:
                    if sent_words not in self.stopWords and sent_words in self.glove:
                        sims.append(self.sim_to_logprob(self.glove.similarity(sent_words, word)))
                    # TODO for OOVs, should we do anything?
                    # elif sent_words not in self.stopWords and sent_word not in self.glove:
                ngram = self.lm.score(word)
                if sims == []:
                    sims = [0.0]
                return (0.5 * self.mean(sims)) + (0.5*ngram)
            else:
                # When swapped word is an OOV
                ngram = self.lm.score(word)
                return (0.5*self.sim_to_logprob(0.0)) + (0.5*ngram)
        elif self.select_mode == '--comabs':
            sims = []
            if word in self.glove:
                for sent_words in sentence:
                    if sent_words not in self.stopWords and sent_words in self.glove:
                        sims.append(self.sim_to_logprob(abs(self.glove.similarity(sent_words, word))))
                    # TODO for OOVs, should we do anything?
                    # elif sent_words not in self.stopWords and sent_word not in self.glove:
                ngram = self.lm.score(word)
                if sims == []:
                    sims = [0.0]
                return (0.5 * self.mean(sims)) + (0.5*ngram)
            else:
                # When swapped word is an OOV
                ngram = self.lm.score(word)
                return (0.5*self.sim_to_logprob(0.0)) + (0.5*ngram)


    def ap(self, good_list, guess_list):
        """
        Change in recall is 1 / good_list iff len(guess) >= len(good)
        Otherwise it's 1 / guess_list
        Formula = p@1 * change in recall @ 1 + p@2 * change in recall at 2
        :param good_list: 
        :param guess_list: 
        :return: 
        """
        # print('guess', guess_list)
        # print('good', good_list)
        good_list = list(good_list)
        # print("checking lengths", 'guess', len(guess_list), 'good', len(good_list))
        # if len(guess_list) >= len(good_list):
        #     ap = self.p_at_x(good_list, guess_list, good_list)
        # else:
        #     ap = self.p_at_x(good_list, guess_list, guess_list)
        print("Checking AP with\n",guess_list, '\n', good_list)
        ap = self.p_at_x(good_list, guess_list, guess_list)
        return ap

    def p_at_x(self, golds, guesses, rec_diff):
        # now rec_diff = guesses
        # TODO there's an issue with the indexing---we're not seeing things that are plainly in the golds list
        print("Starting AP")
        precisions = []
        print("###RECDIFF###", "1 /", len(rec_diff))
        for i in range(len(rec_diff)):
            rec_change = 1 / len(rec_diff)
            sum_at_i = 0
            # print("### checking if p@x is good", guesses[i][0], guesses[i][0] in golds)
            print("### checking if p@x is good", guesses[i], guesses[i] in golds)
            # if guesses[i][0] in golds:
            if guesses[i] in golds:
                zero_out = 1
            else:
                zero_out = 0
            for n in range(i + 1):
                # print(golds)
                # print(guesses)
                print("\tzero_out is", zero_out)
                # print("\t###comparing###", guesses[n][0], guesses[n][0] in golds)
                print("\t###comparing###", guesses[n], guesses[n] in golds)
                print("\tguesses at n:", guesses[:n + 1])
                for guess in guesses[:n + 1]:
                    # if guess in golds[:i+1]:
                    # print("###DOUBLECHECKING guess[0] in golds", guess, guess[0] in golds)
                    # print("###DOUBLECHECKING guess[0] in golds", guess, guess in golds)
                    # if guess[0] in golds:
                    if guess in golds:
                        sum_at_i += 1
                # print("### ARE WE RIGHT AT K?", guesses[i][0], i , guesses[i][0] in golds)
                print("\t### ARE WE RIGHT AT K?", guesses[i], i , guesses[i] in golds)
            # + 1 since range starts at 0
            # print("sum_at_i", sum_at_i)
            # print("i + 1", i + 1)
            prec = sum_at_i / (i + 1)
            # print("at", i, "prec, zero_out", prec, zero_out)
            prec = prec * zero_out
            precisions.append(prec)
            print("\t ap at i:", 'i =', i+1, prec)
        sum_prec = sum(precisions)
        prec_at_x = sum_prec / len(rec_diff)
        print('Total AP:', prec_at_x)
        # pdb.set_trace()
        return prec_at_x



    def select_combo(self, source, index, sentence, cluster, dialog):
        if len(cluster[2]) > 1:
            print("Found one or more for '", source, "'")
            print("Using combo")
            # remove identical entries:
            for key in cluster[2]:
                # do we dare a while loop?
                while source in cluster[2][key]:
                    print("\t removing", source, "from", key)
                    cluster[2][key].pop(cluster[2][key].index(source))
                assert(source not in cluster[2][key])
            print("len of clust", len(cluster[2]))#cluster[2],
            avg_comb = {}
            # assume 4 gram
            # no, assume whole sentence
            # base_score = self.sent_scorer(sentence, sentence[index])
            context_pre = sentence[0:index]
            # context_pre = context_pre[-4:]
            context_post = sentence[index + 1:]
            # context_post = context_post[0:5]
            context = dialog + ' ' + ' '.join(sentence)
            context = context.split()
            base_prob = self.score(' '.join(context))
            print('cluster')
            print("\t", cluster[1:3], '...')
            for key in cluster[2]:
                print("\t starting cluster", key)
                probs = []
                sims = []
                if cluster[2][key] != []:
                    for word in cluster[2][key]:
                        # average probs
                        word_and_context = context_pre + [word] + context_post
                        word_and_context = ' '.join(word_and_context)
                        word_and_context = dialog +' ' + word_and_context
                        prob = base_prob - self.score(word_and_context)
                        probs.append(prob)
                        for cont_word in context:
                            if cont_word not in self.stopWords:
                                    if word in self.glove and cont_word in self.glove:
                                        if word != cont_word:
                                            if self.select_mode in ['--combo', '--com5', '--compos']:
                                                sim = self.glove.similarity(cont_word, word)
                                                sims.append(sim)
                                            elif self.select_mode == '--comabs':
                                                sim = abs(self.glove.similarity(cont_word, word))
                                                sims.append(sim)
                                    else:
                                            sims.append(float(0))
                    # if len(cluster[2]) == 7:    
                        # pdb.set_trace()
                    if self.select_mode in ['--combo', '--com5', '--compos']:
                        if self.select_mode == '--com5':
                            sims = self.top_5(sims)
                        if self.select_mode == '--compos':
                            sims = self.no_negs(sims)
                        # print('sims', sims)
                        average_norm_sim = (self.mean(sims) + 1) / 2
                    elif self.select_mode == '--comabs':
                        average_norm_sim = self.mean(sims)
                        # special care of OOVs in comabs
                        if average_norm_sim == 0.0:
                            average_norm_sim = 0.000001
                    # TODO do we need this check?
                    errormsg = "average_norm_sim is " + str(average_norm_sim) + '\nsims: ' + str(sims)
                    assert(average_norm_sim > 0 and average_norm_sim < 1), errormsg
                    avg_prob = self.mean(probs)
                    avg_comb[key] = (0.5 * math.log(average_norm_sim)) + (0.5 * avg_prob)
                else:
                    print(key, "is empty after removing", source)
            print('KEYS')
            for key in avg_comb:
                print('\t', key, len(cluster[2][key]), "total items")
                for word in cluster[2][key]:
                    print("\t\t", word)
                print('\t', avg_comb[key])
            # omg, max(dict) doesn't work?
            # stackoverflow hack: http://stackoverflow.com/questions/26871866/print-highest-value-in-dict-with-key
            top = max(avg_comb, key=avg_comb.get)
            print("\tmax", top, avg_comb[top])
            return top, avg_comb[top]
        else:
            print("Singleton returned or is a stopword", source, len(cluster))
            return '1', 0.0
    
    
    def select_glove(self, source, index, sentence, cluster, dialog):
        if len(cluster[2]) > 1:
            print("Found one or more for '", source, "'")
            # remove identical entries:
            for key in cluster[2]:
                # do we dare a while loop?
                while source in cluster[2][key]:
                    print("\t removing", source, "from", key)
                    cluster[2][key].pop(cluster[2][key].index(source))
                assert(source not in cluster[2][key])
            print("len of clust", len(cluster[2]))#cluster[2],
            avg_sim = {}
            # computing average sim to original word accorss cluster
            # orig_word = sentence[index]
            context = dialog + ' ' + ' '.join(sentence)
            context = context.split()
            print('cluster')
            print("\t", cluster[1:3], '...')
            for key in cluster[2]:
                print("\t starting cluster", key)
                sims = []
                if cluster[2][key] != []:
                    for cont_word in context:
                        if cont_word not in self.stopWords:
                            for word in cluster[2][key]:
                                if word in self.glove and cont_word in self.glove:
                                    if word != cont_word:
                                        if self.select_mode in ['--glove', '--glo5', '--glopos']:
                                            sim = self.glove.similarity(cont_word, word)
                                            sims.append(sim)
                                        elif self.select_mode == '--glabs':
                                            sim = abs(self.glove.similarity(cont_word, word))
                                            sims.append(sim)
                                else:
                                    sims.append(float(0))
                    if self.select_mode == '--glo5':
                        sims = self.top_5(sims)
                    elif self.select_mode == '--glopos':
                        sims = self.no_negs(sims)
                    avg = self.mean(sims)
                    avg_sim[key] = avg
                else:
                    print(key, "is empty after removing", source)
            print('KEYS')
            for key in avg_sim:
                print('\t', key, len(cluster[2][key]), "total items")
                for word in cluster[2][key]:
                    print("\t\t", word)
                print('\t', avg_sim[key])
            # omg, max(dict) doesn't work?
            # stackoverflow hack: http://stackoverflow.com/questions/26871866/print-highest-value-in-dict-with-key
            top = max(avg_sim, key=avg_sim.get)
            print("\tmax", top, avg_sim[top])
            return top, avg_sim[top]
        else:
            print("Singleton returned or is a stopword", source, len(cluster))
            return '1', 0.0
        # pass

    def no_negs(self, sim_array):
        """
        remove all negative cos sims from an array
        """
        allpos = []
        for sim in sim_array:
            if sim > 0.0:
                allpos.append(sim)
        assert(len(allpos) <= len(sim_array))
        # in case there are no positive values
        if allpos == []:
            allpos.append(0.0)
        return allpos

    def top_5(self, sim_array):
        """
        only keep the 5 most positive cos sims
        """
        top5 = []
        # print("top5 sim array", sim_array)
        for i in range(5):
            if sim_array != []:
                top = max(sim_array)
                top5.append(top)
                sim_array.pop(sim_array.index(top))
        assert(len(top5) <= 5)
        return top5


    def select_pmi(self, source, index, sentence, cluster, dialog):
        get_pmi = lambda x, y: (self.score(x + ' ' + y) - (self.score(x) + self.score(y))) / math.log(2)
        if len(cluster[2]) > 1:
            print("Found one or more for '", source, "'")
            # remove identical entries:
            for key in cluster[2]:
                # do we dare a while loop?
                while source in cluster[2][key]:
                    print("\t removing", source, "from", key)
                    cluster[2][key].pop(cluster[2][key].index(source))
                assert(source not in cluster[2][key])
            print("len of clust", len(cluster[2]))#cluster[2],
            avg_pmi = {}
            # assume 4 gram
            # no, assume whole sentence
            # for single word sentences
            # TODO delete?
            if len(sentence) == 1:
                context_post = ['']
                context_pre = ['']
            else:
                context_pre = sentence[0:index]
                # context_pre = context_pre[-4:]
                context_post = sentence[index + 1:]
                # context_post = context_post[0:5]
            print('clusters')
            print(cluster)
            assert(len(cluster) == 3)
            for c in cluster[2]:
                # print("c is ", c)
                print("\t", c, cluster[2][c][:3], '...')
            for key in cluster[2]:
                print("\t starting cluster", key)
                pmis_per_word = []
                if cluster[2][key] != []:
                    for word in cluster[2][key]:
                        # word_and_context = context_pre + [word] + context_post
                        for cont_word in context_pre:
                            pmi = get_pmi(cont_word, word)
                            print("context, word", cont_word, word, pmi)
                            pmis_per_word.append(pmi)
                            # pmis_per_word.append(get_pmi(cont_word, word))
                        for cont_word in context_post:
                            pmi = get_pmi(word, cont_word)
                            print("word, context", word, cont_word, pmi)
                            pmis_per_word.append(pmi)
                            # pmis_per_word.append(get_pmi(word, cont_word))
                        # pdb.set_trace()
                        # prob = self.score(' '.join(word_and_context))
                        # print("\tword", word)
                        # print("\t\tprob",prob)
                        # probs.append(prob)
                    print("pmis_per_word", pmis_per_word)
                    avg = self.mean(pmis_per_word)
                    # pdb.set_trace()
                    avg_pmi[key] = avg
                else:
                    print(key, "is empty after removing", source)
            # print('CONTEXT', context)
            print('KEYS')
            # TODO do we still need max? can we averate pmis like this?
            # print(avg_prob)
            for key in avg_pmi:
                print('\t', key, len(cluster[2][key]), "total items")
                for word in cluster[2][key]:
                    print("\t\t", word)
                print('\t', avg_pmi[key])
            # omg, max(dict) doesn't work?
            # stackoverflow hack: http://stackoverflow.com/questions/26871866/print-highest-value-in-dict-with-key
            top = max(avg_pmi, key=avg_pmi.get)
            print("\tmax", top, avg_pmi[top])
            return top, avg_pmi[top]
        else:
            print("Singleton returned or is a stopword", source, len(cluster))
            return '1', 0.0

    def select(self, source, index, sentence, cluster, dialog):
        if len(cluster[2]) > 1:
            print("Found one or more for '", source, "'")
            # remove identical entries:
            for key in cluster[2]:
                # do we dare a while loop?
                while source in cluster[2][key]:
                    print("\t removing", source, "from", key)
                    cluster[2][key].pop(cluster[2][key].index(source))
                assert(source not in cluster[2][key])
            print("len of clust", len(cluster[2]))#cluster[2],
            avg_prob = {}
            # assume 4 gram
            # no, assume whole sentence
            context_pre = sentence[0:index]
            # context_pre = context_pre[-4:]
            context_post = sentence[index + 1:]
            # context_post = context_post[0:5]
            print('cluster')
            print("\t", cluster[1:3], '...')
            for key in cluster[2]:
                print("\t starting cluster", key)
                probs = []
                if cluster[2][key] != []:
                    for word in cluster[2][key]:
                        word_and_context = context_pre + [word] + context_post
                        word_and_context = ' '.join(word_and_context)
                        word_and_context = dialog +' ' + word_and_context
                        prob = self.sent_scorer(word_and_context, word)
                        print("\tword: ", word)
                        print("\t\tprob: ",prob)
                        probs.append(prob)
                    avg = self.mean(probs)
                    avg_prob[key] = avg
                else:
                    print(key, "is empty after removing", source)
            # print('CONTEXT', context)
            print('KEYS')
            # print(avg_prob)
            for key in avg_prob:
                print('\t', key, len(cluster[2][key]), "total items")
                for word in cluster[2][key]:
                    print("\t\t", word)
                print('\t', avg_prob[key])
            # omg, max(dict) doesn't work?
            # stackoverflow hack: http://stackoverflow.com/questions/26871866/print-highest-value-in-dict-with-key
            top = max(avg_prob, key=avg_prob.get)
            print("\tmax", top, avg_prob[top])
            return top, avg_prob[top]
        else:
            print("Singleton returned or is a stopword", source, len(cluster))
            return '1', 0.0

    def lex_swap(self, sentence, cluster, label, words, dialog):
        # sentence = sentence.split()
        print("Using semantic clustering")
        print("sentence", sentence)
        print("total clusters", len(cluster))
        assert(len(sentence) == len(cluster))
        print('total', len(words))
        chosen_clusters = []
        for i in range(len(cluster)):
            # print('sentence[i]', sentence[i])
            # print('cluster[i][1]', cluster[i][1])
            # print('cluster[i]', cluster[i])
            # TODO is there a better check for this now that we're using stopwords?
            # assert(sentence[i].lower() == cluster[i][1])
            if self.mode == '--normal':
                if self.select_mode in ['--ngram', '--ngram-word']:
                    symset, percentage = self.select(sentence[i], i, sentence, cluster[i], dialog)
                elif self.select_mode == '--pmi':
                    symset, percentage = self.select_pmi(sentence[i], i, sentence, cluster[i], dialog)
                elif self.select_mode in ['--glove', '--glabs', '--glo5', '--glopos']:
                    symset, percentage = self.select_glove(sentence[i], i, sentence, cluster[i], dialog)
                elif self.select_mode in ['--combo', '--comabs', '--com5', '--compos']:
                    symset, percentage = self.select_combo(sentence[i], i, sentence, cluster[i], dialog)
                else:
                    sys.exit("selection method can only be '--ngram', '--glove', '--glabs', '--glo5', '--glopos', '--combo', '--comabs', '--com5', '--compos',  or '--pmi'")
                chosen_clusters.append(cluster[i][2][symset])
            elif self.mode == '--all':
                allclust = []
                for symset in cluster[i][2]:
                    for word in cluster[i][2][symset]:
                        allclust.append(word)
                chosen_clusters.append(allclust)
            elif self.mode == '--first':
                # TODO change this to picking the semclust with the highest unigram score
                chosen_clusters.append(cluster[i][2]['1'])
            # symset = '1'
            # self.replace(sentence, i, cluster[i][2][symset], words, 'semclusters', label, percentage)
        print("chosen", chosen_clusters, len(chosen_clusters))
        print("len sent", len(sent))
        assert(len(chosen_clusters) == len(sentence))
        return chosen_clusters

    def swap_wordnet(self, tagged, sentence, label, words, dialog):
        """
        Same as lex_swap, but for wordnet
        :param tagged:[(word, pos_tag), (...)]
        :param sentence:
        :param label:
        :param words:
        :return:
        """
        print("Using wordnet")
        print("\tsentence", sentence)
        print("\ttagged", tagged)
        # analogous to lex_swap cluster
        hypo_clusters = []
        hyper_clusters = []
        # syn_clusters = []
        index = 0
        for word in tagged:
            hypo_cluster = []
            hyper_cluster = []
            # TODO not the best, ignores synsets
            # syn_cluster = []
            if word[0] not in self.stopWords:
                # from wordnet's nltk source code:
                # #{ Part-of-speech constants
                #  ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
                token = word
                # TODO source code was inaccurate---this is now the nltk POS tag set
                if word[1][0] == 'J':
                    pos = 'a'
                elif word[1][0] == 'R':
                    pos = 'r'
                elif word[1][0] in ['N', 'D']:
                    pos = 'n'
                elif word[1][0] == 'V':
                    pos = 'v'
                else:
                    print("\tWHY DIDN'T WE GET A POS TAG???", word)
                    pos = 'n'
                lemma = self.wnl.lemmatize(word[0], pos=pos)
                # TODO do we need to check for the correct sense?
                senses = {}
                # sense methods pulled from dir(wordnet.synsets('have')[2])
                # these don't work and are already covered by hypernums
                # this was an idea to get synonyms
                # TODO clean this up
                # try:
                #     for name in wordnet.synsets(lemma)[0].hypernyms()[0].lemma_names():
                #         syn_cluster.append(name)
                # except:
                #     pdb.set_trace()
                for sense in wordnet.synsets(lemma):
                    if sense.pos() == pos:
                        sense_num = sense.name()[-2:]
                        sense_pos = sense.pos()
                        print('\tpos =', pos, 'possible sense =', sense_num)
                        sense_clust = []
                        print('\t\tbuilding wordnet sysnsets with query', sense.name())
                        synset = wordnet.synset(sense.name())
                        print('\t\tresutls were', synset)
                        for hypo in synset.hyponyms():
                            sense_clust.append(hypo.name().split('.')[0])
                        for hyper in synset.hypernyms():
                            sense_clust.append(hyper.name().split('.')[0])
                        if sense_clust != []:
                            senses[sense_num] = sense_clust
                print('\tFinal senses lists:', senses.keys())
                queries = []
                if len(senses) > 1:
                    print("\tFinding the correct synset from above with:\n\t", sentence[index], index)
                    print("\t", sentence)
                    for poss_sense in senses:
                        print("\t", poss_sense)
                        for word in senses[poss_sense]:
                            print('\t\t', word)
                    if self.mode == '--normal':
                        if self.select_mode in ['--ngram', '--ngram-word']:
                            sense, percentage = self.select(sentence[index], index, sentence, [lemma, pos, senses], dialog)
                        elif self.select_mode == '--pmi':
                            sense, percentage = self.select_pmi(sentence[index], index, sentence, [lemma, pos, senses], dialog)
                        elif self.select_mode in ['--glove', '--glabs', '--glo5', '--glopos']:
                            sense, percentage = self.select_glove(sentence[index], index, sentence, [lemma, pos, senses], dialog)
                        elif self.select_mode in ['--combo', '--comabs', '--com5', '--compos']:
                            sense, percentage = self.select_combo(sentence[index], index, sentence, [lemma, pos, senses], dialog)
                        else:
                            sys.exit("selection method can only be '--ngram', '--glove', '--glabs', '--glo5', '--glopos', '--combo', '--comabs', '--com5', '--compos',  or '--pmi'")
                        queries.append('.'.join([lemma, pos, sense]))
                    elif self.mode == '--all':
                        for s in senses:
                            queries.append('.'.join([lemma, pos, s]))
                    elif self.mode == '--first':
                        queries.append('.'.join([lemma, pos, '01']))
                    print('\tsense chosen', sense)
                elif len(senses) == 1:
                    print("\tOnly one found", senses)
                    one = list(senses.keys())
                    percentage = 0.0
                    assert(len(one) == 1)
                    queries.append('.'.join([lemma, pos, one[0]]))
                else:
                    print("\tpossible error with sense?", senses)
                    sense = "01"
                    percentage = 0.0
                    queries.append('.'.join([lemma, pos, sense]))
                # >>> test = wordnet.synset('drug.n.01')
                # query = '.'.join([lemma, pos, sense])
                for query in queries:
                    print('\ttrying', query)
                    try:
                        synset = wordnet.synset(query)
                        print('\tquery hyponyms')
                        for hypo in synset.hyponyms():
                            print('\t\t', hypo)
                        print('\tquery hypernyms')
                        for hyper in synset.hypernyms():
                            print('\t\t', hyper)
                # >>> test.hyponyms()
                # [Synset('abortifacient.n.01'), Synset('agonist.n.04'), Synset('anesthetic.n.01'), Synset('antagonist.n.03'), Synset('anti-tnf_compound.n.01'), Synset('antisyphilitic.n.01'), Synset('arsenical.n.01'), Synset('botanical.n.01'), Synset('brand-name_drug.n.01'), Synset('controlled_substance.n.01'), Synset('dilator.n.02'), Synset('diuretic_drug.n.01'), Synset('drug_of_abuse.n.01'), Synset('feosol.n.01'), Synset('fergon.n.01'), Synset('fertility_drug.n.01'), Synset('generic_drug.n.01'), Synset('intoxicant.n.02'), Synset('levallorphan.n.01'), Synset('medicine.n.02'), Synset('miotic_drug.n.01'), Synset('mydriatic.n.01'), Synset('narcotic.n.01'), Synset('pentoxifylline.n.01'), Synset('psychoactive_drug.n.01'), Synset('psychotropic_agent.n.01'), Synset('relaxant.n.01'), Synset('soporific.n.01'), Synset('stimulant.n.02'), Synset('suppressant.n.01'), Synset('synergist.n.01'), Synset('virility_drug.n.01')]
                # >> > str(test.hyponyms()[0]).split(".")[0].replace("Synset('", "")
                        for hypo in synset.hyponyms():
                            hypo_cluster.append(hypo.name().split('.')[0])
                        for hyper in synset.hypernyms():
                            hyper_cluster.append(hyper.name().split('.')[0])
                        # cluster.append(str(hypo.split(".")[0].replace("Synset('", "")))
                    except:
                        print("\tNo synset found for query:", query, "in wordnet")
            hypo_clusters.append(hypo_cluster)
            hyper_clusters.append(hyper_cluster)
            # syn_clusters.append(syn_cluster)
            index += 1
        assert(len(sentence) == len(hypo_clusters))
        assert(len(sentence) == len(hyper_clusters))
        # assert(len(sentence) == len(syn_clusters))
        # print("###wordnet sentence", sentence)
        return hypo_clusters, hyper_clusters
            # , syn_clusters
        # for i in range(len(sentence)):
            # TODO verify that these are actually missing in wordnet
            # print('\tsentence[', i, ']', sentence[i])
            # print('\tclusters[', i, ']', hypo_clusters[i])
            # print('\tclusters[', i, ']', hyper_clusters[i])
            # print('cluster[i]', cluster[i])
            # symset = '1'
            # self.replace(sentence, i, hypo_clusters[i], words, 'wordnet-hypo', label, percentage)
            # self.replace(sentence, i, hyper_clusters[i], words, 'wordnet-hyper', label, percentage)

    def swap_w2v(self, sentence, label, words, dialog):
        print("Using word2vec")
        print("\tsentence", sentence)
        # TODO add ginsim and w2v
        # >>> model.most_similar(['drug'], topn=5)
        # [('drugs', 0.938819169998169), ('methamphetamine', 0.8612675666809082), ('cocaine', 0.8234723806381226), ('marijuana', 0.7992461323738098), ('oxycontin', 0.7919926047325134)]
        clusters = []
        cos_sims = []
        for word in sentence:
            cluster = []
            if word not in self.stopWords:
                # in case of OOVs
                if word in self.model:
                    top5 = self.model.most_similar([word], topn=10)
                    for sim_word in top5:
                        cluster.append(sim_word[0])
                        cos_sims.append(sim_word[1])
            clusters.append(cluster)
        # TODO same code used: convert to function?
        assert(len(sentence) == len(clusters))
        print("w2v clusters:")
        try:
            min = min(cos_sims)
        except:
            min = 0.0
        for c in clusters:
            print("\t", c)
        return clusters
        # for i in range(len(clusters)):
            # print('sentence[i]', sentence[i])
            # print('cluster[i][1]', cluster[i][1])
            # print('cluster[i]', cluster[i])
            # symset = '1'
            # self.replace(sentence, i, clusters[i], words, 'word2vec', label, min)

    def load_lm(self, lmfile):
        print("Loading language model", lmfile)
        # print(lmfile[-5:])
        if lmfile[-4:] == 'arpa':
            models = arpa.loadf(lmfile)
            self.lm = models[0]
            self.score = lambda x: self.lm.log_p(x)
        elif lmfile[-3:] == 'bin':
            self.lm = kenlm.LanguageModel(lmfile)
            self.score = lambda x: self.lm.score(x)
        else:
            sys.exit("I can only read kenlm .bin or .arpa file")

    def add_candidate(self, candidates, new_list):
        # print("candidates", candidates)
        # print("new list", new_list)
        assert(len(candidates) == len(new_list))
        for i in range(len(candidates)):
            for new_word in new_list[i]:
                candidates[i].append(new_word)
                candidates[i] = sorted(set(candidates[i]))
        return candidates

    def rm_dups(self, candidates):
        print("Removing duplicates:")
        for cand in candidates:
            print("\t", cand)
        outlist = []
        for cand_list in candidates:
            cand_list = sorted(set(cand_list))
            outlist.append(cand_list)
        print("Dupes removed")
        for cand in outlist:
            print("\t", cand)
        return outlist

    def rank_paras(self, paraphrases, label_cut):
        """
        Return a list of ranked swaps
        """
        ranked_words = []
        i = 0
        while i < label_cut:
            best = max(paraphrases, key=paraphrases.get)
            ranked_words.append(best[4])
            paraphrases.pop(best)
            i += 1
        return ranked_words 

    def rank_and_prune(self, candidates, sentence, topn):
        # TODO depreciate this? It's still being used, but I'm pretty sure
        # TODO it shouldn't be, though I don't think we can just delete it
        """
        Cprrection, this ranking is surplufuous. It does filter out duplicates
        and non-matching POS tags, but the reranking is left over from an
        earlier iteration, but does not affect the final result.
        """
        # print("candidates:", candidates)
        assert(len(sentence)==len(candidates))
        print("### During rank and prune", sentence)
        print("Mode is", self.mode)
        bestfit = []
        # Clear out the duplicates and filter so that POS tags match
        candidates = self.rm_dups(candidates)
        # print("candidates:", candidates)
        for i in range(len(sentence)):
            cand_scores = {}
            # print("sentence", sentence)
            # base_score = self.lm.score(' '.join(sentence))
            for new_word in candidates[i]:
                sentence_tagged = nltk.pos_tag(sentence)
                testsent = sentence
                orig_word = sentence[i]
                testsent[i] = new_word
                # print(self.lm.score(' '.join(testsent)), 'testsent', ' '.join(testsent))
                # I'm putting these backwards so it's more intuitive for me
                # >> > a = -36.03154373168945
                # >> > b = -31.504854202270508
                # >> > a - b
                # -4.526689529418945
                # >> > b - a
                # 4.526689529418945
                # diff = base_score - self.lm.score(' '.join(testsent))
                # cand_scores[new_word] = self.lm.score(' '.join(testsent))
                # if diff > 0:
                #     pos_scores[new_word] = diff
                testsent_tagged = nltk.pos_tag(testsent)
                # pos and identity check
                if testsent_tagged[i][1] == sentence_tagged[i][1] and testsent_tagged[i][0] != sentence_tagged[i][0]:
                    print('### FOUND ONE', testsent_tagged[i], sentence_tagged[i])
                    score = self.lm.score(' '.join(testsent))
                    cand_scores[testsent[i]] = score
                # else:
                #     print('###NOPE', testsent_tagged[i], sentence_tagged[i], sentence_tagged)
                # elif diff < 0:
                #     neg_scores[new_word] = diff
                sentence[i] = orig_word
                # print("###during-2 ranking", sentence)
            print("BEFORE PRUNNING")
            for cs in cand_scores:
                print("\t", cs)
            best_for_i = self.prune(cand_scores, topn)
            # print('pos_scores', pos_scores)
            # print('neg_scores', neg_scores)
            print("BEST SCORERS!", best_for_i)
            # print("reordered", self.reorder(sentence, i, best_for_i))
            bestfit.append(best_for_i)
        assert(len(sentence) == len(bestfit))
        return bestfit

    def reorder(self, sentence, i, wordlist):
        rankedlist = []
        rankeddict = {}
        # print("###sentence in reorder", sentence)
        for word in wordlist:
            orig_word = sentence[i]
            testsent = sentence
            testsent[i] = word
            # pos and identity check
            score = self.lm.score(' '.join(testsent))
            rankeddict[word] = score
            sentence[i] = orig_word
        # print("###sentence in reorder-2", sentence)
        for n in range(len(rankeddict)):
            best = max(rankeddict, key=rankeddict.get)
            rankedlist.append((best, rankeddict[best]))
            rankeddict.pop(best)
        return rankedlist


    def prune(self, cand_dict, topn):
        tops = []
        # print("start while loop. len(tops)", len(tops))
        for i in range(topn):
            if cand_dict != {}:
                best = max(cand_dict, key=cand_dict.get)
                tops.append((best, cand_dict[best]))
                cand_dict.pop(best)
                # print("looking at pos. len(tops)", len(tops))
            # elif neg_dict != {}:
            #     print('neg_dict', neg_dict)
                # best = min(neg_dict, key=neg_dict.get)
                # tops.append(best)
                # neg_dict.pop(best)
                # print("pos is empty, checking neg. len(tops)", len(tops))
        assert(len(tops) <= topn)
        return tops

    def remove_dups(self, candidates):
        out_cand = []
        for cand in candidates:
            out_cand.append(sorted(set(cand)))
        return out_cand

    def sort_by_value(self, paraphrases):
        """Convert arrays to dictionaries with diff score as the value"""
        para_by_value = {}
        for label in paraphrases:
            para_by_value[label] = {}
            for para in paraphrases[label]:
                para_by_value[label][para] = para[-1]
                # pdb.set_trace()
        for label in para_by_value:
            # print("label", label)
            errormsg = str(len(sorted(set(para_by_value[label])))) + " != " + str(len(sorted(set(paraphrases[label]))))
            assert(len(sorted(set(para_by_value[label]))) == len(sorted(set(paraphrases[label])))), errormsg
        # pdb.set_trace()
        # Now sort
        # new_paras = {}
        # for label in para_by_value:
        #     para_sorted = sorted(para_by_value[label].items(), key=operator.itemgetter(1))
        #     new_paras[label] = para_sorted
            # pdb.set_trace()
        # for label in new_paras:
        #     errormsg = str(len(sorted(set(new_paras[label])))) + " != " + str(len(sorted(set(paraphrases[label]))))
        #     assert(len(sorted(set(new_paras[label]))) == len(sorted(set(paraphrases[label])))), errormsg
        # pdb.set_trace()
        # return new_paras
        return para_by_value

    def pullmin(self, sorted_dict, num):
        """
        data structure:
            [(('sent', source, target, newsent, logprob diff), logprobdiff), ...]
        :param sorted_dict: 
        :param num: 
        :return: 
        """
        returnable = {}
        for label in sorted_dict:
            returnable[label] = []
            for i in range(num):
                if len(sorted_dict[label]) != 0:
                    # best = max(sorted_dict[label], key=sorted_dict[label].get)
                    best = min(sorted_dict[label], key=sorted_dict[label].get)
                    returnable[label].append(best)
                    sorted_dict[label].pop(best)
            assert(len(returnable[label]) <= num)
        assert(len(returnable) == len(sorted_dict))
        # pdb.set_trace()
        return returnable, sorted_dict



if __name__ == '__main__':
    # TODO this got out of hand. Please clean it up. Please?!
    # separate each work:
    if len(sys.argv) < 8:
        print('len(sys.argv)', len(sys.argv))
        print("sys.argv:", sys.argv)
        print("please run the system like so:")
        print("python3 lexsub.py paraphrases.pkl output.tsv -lm lm.kenlm.bin -c --wordnet --w2v vectors.bin --normal --pmi --para 5 10")
        sys.exit()
    # label = ' '.join(sys.argv[3:])
    l = lexsub(sys.argv[2])
    l.load_paraphrases(sys.argv[1])
    print('looking at options', sys.argv[3:])
    ####################
    # OPTIONS SECTIONS #
    ####################
    for option in sys.argv[3:]:
        # print('testing if wordnet ==')
        # print('\t-c', option == '-c')
        # print('\t--wordnet', option == '--wordnet')
        # print('\t--w2v', option == '--w2v')
        cutoff = int(sys.argv[-1])
        label_cut = int(sys.argv[-2])
        if option == '-c':
            l.sem_clust = True
            print("option", option)
        elif option == '--wordnet':
            l.wordnet = True
            print("Wordnet is on")
        elif option == '--w2v':
            index = sys.argv.index(option)
            l.w2v = True
            print("Loading option from index", index, "and file from index", index+1)
            print("option", option)
            print("file", sys.argv[index+1])
            l.load_w2v(sys.argv[index + 1])
        elif option == '-lm':
            index = sys.argv.index(option)
            print("Loading option from index", index, "and file from index", index+1)
            print("option", option)
            print("file", sys.argv[index+1])
            l.load_lm(sys.argv[index + 1])
        elif option in ['--label', '--para']:
            if option == '--label':
                do_labels = True
            elif option == '--para':
                do_labels = False
        elif option in ['--normal', '--all', '--first']:
            l.mode = option
            print("System mode:", option)
        elif option in ['--pmi', 
                        '--ngram', 
                        '--ngram-word',
                        '--glove', 
                        '--glabs',
                        '--glo5',
                        '--glopos',
                        '--combo', 
                        '--comabs',
                        '--com5',
                        '--compos']:
            l.select_mode = option
            if option in ['--glove', '--glabs', '--glo5', '--glopos',
                          '--combo', '--comabs', '--com5', '--compos']:
                index = sys.argv.index(option)
                l.load_glove(sys.argv[index + 1]) 
            print('selection mode:', l.select_mode) 
    # print("label", label)
    # print("label in l.paraphrases?", label in l.paraphrases)
    ########################################
    # GENERATE AND COLLECT THE PARAPHRASES #
    ########################################
    para_by_label = {}
    for label in l.paraphrases:
        # adding param for threshold
        # if len(l.paraphrases[label]) > cutoff:
        # l.paraphrases[label] is an entire dialog. We're only interested in dialog[0]
        if len(l.paraphrases[label]) < cutoff:
            para_by_label[label] = []
            words = l.get_words(l.paraphrases[label])
            print("Total variant vocab:", len(words))
            if do_labels:
                data = [(label, [label])]
            else:
                data = l.paraphrases[label]
            for dialog in data:
                sent = dialog[0]
                prev_turns = ' '.join(dialog[1])
                tagged = l.breaksent(sent)
                # l.pull_diffs(l.orig, l.paraphrases)
                # print(tagged)
                sentence = []
                candidates = []
                for postagged in tagged:
                    candidates.append([])
                    sentence.append(postagged[0])
                if l.sem_clust:
                    clusters = []
                    for word in tagged:
                        clusters.append(l.query(word[0].lower(), word[1]))
                    print("SENTENCE", sentence)
                    print('CLUSTERS')
                    for c in clusters:
                        print('\t', c[0], c[1])
                        for sense in c[2]:
                            print('\t\t', sense, "length", len(c[2][sense]), c[2][sense][0:3], '...')
                    pptb_clust = l.lex_swap(sentence, clusters, label, words, prev_turns)
                    # print("CURRENTLY IN LEX_SWAP")
                    candidates = l.add_candidate(candidates, pptb_clust)
                # print('wordnet', l.wordnet)
                if l.wordnet:
                    hypo_clust, hyper_clust = l.swap_wordnet(tagged, sentence, label, words, prev_turns)
                    # print("CURRENTLY IN WORDNET")
                    # print('hyper', hyper_clust)
                    # print('hypo', hypo_clust)
                    candidates = l.add_candidate(candidates, hyper_clust)
                    candidates = l.add_candidate(candidates, hypo_clust)
                    # candidates = l.add_candidate(candidates, syn_clust)
                if l.w2v:
                    vec_clust = l.swap_w2v(sentence, label, words, prev_turns)
                    # print("CURRENTLY IN WORD2VEC")
                    candidates = l.add_candidate(candidates, vec_clust)
                # print("### pre-ranking", sentence)

                candidates = l.remove_dups(candidates)
                # TODO depreciate 50 here = top-n
                ranked = l.rank_and_prune(candidates, sentence, 100)
                print("### post-ranking", sentence)
                print("ranked")
                for r in ranked:
                    print("\t", r)
                print("sentence", sentence)
                for i in range(len(sentence)):
                    print("Starting at i", i, sentence[i])
                    if ranked[i] != []:
                        print("###before replacing", sentence)
                        paraphrases = l.replace(sentence, i, ranked[i], words, label)
                        for paras in paraphrases:
                            para_by_label[label].append(paras)
                        # print("CHECKING PRECISION")
                        # ap = l.ap(words, ranked[i])
                        # l.kept_map.append(ap)
                        # print('AP', ap)
                        # def replace(self, sentArray, index, cluster, words, label):
    ######################
    # NOW PRUNE AND RANK #
    ######################
    sorted_paraphrases = l.sort_by_value(para_by_label)
    chosen, orig_sorted = l.pullmin(sorted_paraphrases, label_cut)
    for label in chosen:
        words = l.get_words(l.paraphrases[label])
        ranked = []
        for sent in chosen[label]:
            outline = []
            for tupe in sent:
                outline.append(str(tupe))
            outline.append('\n')
            ranked.append(sent[3])
            l.outcsv.write('\t'.join(outline))
            l.kept += 1
        print('pre-ap ranked', ranked)
        # TODO Why are we getting 0.0 for MAP... I think that shouldn't be happening.
        if ranked != []:
            ap = l.ap(words, ranked)
            l.kept_map.append(ap)
            print("AP", ap)
        # pdb.set_trace()
    # TEMPORARY TODO DELETE WHEN NOT NEEDED
    # tempout = open(l.didnt_make_it, 'w')
    tempout = l.didnt_make_it
    for label in orig_sorted:
        for sent in orig_sorted[label]:
            l.rejected += 1
            outline = []
            for cell in sent[0]:
                outline.append(str(cell))
            outline.append("\n")
            tempout.write('\t'.join(outline))
    # pdb.set_trace()
    # TODO, move eval metrics and pruning conditions here
    # print("precision:", l.prec_num / l.prec_den)
    print("kept paraphrased", l.kept)
    print("rejected paraphrases", l.rejected)
    print("total generated", l.kept + l.rejected)
    print("kept mean average precision", l.mean(l.kept_map))
