#!/usr/bin/env python3

import pandas as pd
# from tqdm import tqdm_notebook
from tqdm import tqdm
import pdb

import glob
import logging
import os
import random
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import random
from torch.utils.data.distributed import DistributedSampler
# from tqdm import tqdm_notebook, trange
from tqdm import tqdm, trange

from sklearn.metrics import mean_squared_error, matthews_corrcoef, confusion_matrix
from scipy.stats import pearsonr

from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer,
                                  XLMConfig, XLMForSequenceClassification, XLMTokenizer, 
                                  XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
                                  RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

from pytorch_transformers import AdamW, WarmupLinearSchedule



from utils import (convert_examples_to_features,
                   output_modes, processors)

#########################################################
# Import script 'general' from 'data scripts' directory #
# in m3d repo. You may need to change this path         #
#########################################################
import sys
sys.path.insert(1, '../../data scripts')
from general import Collect

class Classifier:
    def __init__(self, bert_model, device, checkpoint):
        MODEL_CLASSES = {
            'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
            'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
            'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
            'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
        }

        self.output_mode = 'classification'

        task_name = 'binary'

        config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']

        self.config = config_class.from_pretrained(checkpoint,
                                              num_labels=2,
                                              finetuning_task=task_name
                                              )

        self.tokenizer = tokenizer_class.from_pretrained(bert_model)

        # self.bert = model_class.from_pretrained(model)
        self.model = model_class.from_pretrained(checkpoint)

        # torch.distributed.init_process_group(backend='nccl', world_size=2)
        # model = nn.parallel.DistributedDataParallel(model, device_ids=[0,1], output_device=0)

        # model = nn.DataParallel(model)
        # self.bert.to(device)
        self.model.to(device)

        self.device = device

        task = task_name

        self.processor = processors[task]()
        label_list = self.processor.get_labels()
        num_labels = len(label_list)

        self.collect = Collect()

    def load_examples(self, string_list):
        # processor = processors[task]()
        # output_mode = args['output_mode']
        label_list = self.processor.get_labels()
        examples = self.processor._create_examples(string_list, 'dev')

        # print(len(examples))
        # # examples  = [example for example in examples if np.random.rand() < undersample_scale_factor]
        # examples  = [example for example in examples]
        # print(len(examples))
        max_seq_length = 512

        features = convert_examples_to_features(examples, label_list,
            max_seq_length, self.tokenizer, self.output_mode,
            cls_token_at_end=bool('bert' in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=self.tokenizer.cls_token,
            sep_token=self.tokenizer.sep_token,
            cls_token_segment_id=2 if 'bert' in ['xlnet'] else 0,
            pad_on_left=bool('bert' in ['xlnet']),                 # pad on the left for xlnet
            pad_token_segment_id=4 if 'bert' in ['xlnet'] else 0,
            process_count=2)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        if self.output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        elif self.output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        return dataset

    def standardize(self, datas):
        standard_data = []

        guid = 0
        faux_label = '0'
        # Because someone hacked up the data processor
        # Behold, a vestigial organ
        the_letter_a = 'a'
       
        stop = 0
        for datum in tqdm(datas):
            # Starts with http and isn't a sentence with an URL in it
            if datum.startswith("http") and \
               len(datum.split(' ')) == 1 and \
               datum[-4:] not in ['.mp3', '.mp4']:
                text = self.collect.get_text(datum)
                standard_data.append([guid, faux_label, the_letter_a, text])
            else:
                standard_data.append([guid, faux_label, the_letter_a, datum])
            guid += 1

        # Naw, this is too complex
        # try:
        #     response = requests.get("http://www.avalidurl.com/")
        #     print("URL is valid and exists on the internet")
        # except requests.ConnectionError as exception:
        #     print("URL does not exist on Internet")
        # pdb.set_trace()

        return standard_data

    def inference(self, datas):
        print("Standardizing data")
        standardized_data = self.standardize(datas)
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        print("Loading data into torch")
        eval_dataset = self.load_examples(standardized_data)

        results = {}
        
        print("Starting seq sampler")
        eval_sampler = SequentialSampler(eval_dataset)
        print("running data loader")
        eval_dataloader = DataLoader(eval_dataset,
                                     sampler=eval_sampler,
                                     batch_size=1)

        # Eval!
        print("***** Running evaluation*****")
        print("  Num examples = %d", len(eval_dataset))
        print("  Batch size = %d", 1)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        # for batch in tqdm_notebook(eval_dataloader, desc="Evaluating"):
        print("Starting eval")
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if 'bert' in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                          'labels':         batch[3]}
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        if nb_eval_steps != 0:
            eval_loss = eval_loss / nb_eval_steps

        classes = np.argmax(preds, axis=1)

        return classes, preds

if __name__ == '__main__':
    #############################################################
    # This is just if you want to run it as a standalone script #
    # -DLK                                                      #
    #############################################################
    import argparse

    parser = argparse.ArgumentParser(description='Classification extractor')

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, required=True,
                        help="location of checkpoint folder")
    parser.add_argument("-d", '--data', type=str, required=True,
                        help="data file (plain test). 1 URL or article per line.")
    args = parser.parse_args()

    datas = [x.strip() for x in open(args.data, 'r').readlines()]

    checkpoint = args.checkpoint

    ##########################################################
    # This is how to actually pragrammatically run inference #
    # -DLK                                                   #
    ##########################################################

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'

    print("Loading bert")
    c = Classifier('bert-base-uncased', device, checkpoint)
    # c = Classifier('bert-base-multilingual-uncased', device, checkpoint)
    
    print("Starting inference")
    # try:
    predictions, logits = c.inference(datas)
    # except:
    #     pdb.set_trace()

    ################################
    # Just a printout for pretties #
    # -DLK                         #
    ################################

    for pred, log in zip(predictions, logits):
        print("class prediction: {}\t{}\t{}".format(pred, log[0], log[1]))
