#!/usr/bin/env bash

for i in *.txt
do python3 -m gensim.scripts.glove2word2vec --input $i --output $i.word2vec
done
