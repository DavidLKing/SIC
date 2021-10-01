import sys
import pdb
from os import listdir
from os.path import isfile, join
import pandas as pd
from beto_utils import process_scraped_data


data_dir = 'nov_dec_data/'
out_dir = 'spanish/'
filepaths = [join(data_dir, f) for f in listdir(data_dir) if isfile(join(data_dir, f))]
train_df, val_df, test_df = process_scraped_data(filepaths, separate_domains=True)

def writeout(dataframe, filename):
    with open(filename, 'w') as outfile:
        for anno, text in zip(
                dataframe['labels'].values.tolist(),
                dataframe['text'].values.tolist()
                ):
            outfile.write("{}\t{}\n".format(anno, text))

writeout(train_df, out_dir + 'train_pre.tsv')
writeout(val_df, out_dir + 'dev_pre.tsv')
writeout(test_df, out_dir + 'test_pre.tsv')
        
