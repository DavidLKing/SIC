import pandas as pd
# from tqdm import tqdm_notebook
from tqdm import tqdm
import pdb

prefix = 'data-adapt/'

train_df = pd.read_csv(prefix + 'train_pre.tsv', sep='\t', header=None)
train_df.head()

test_df = pd.read_csv(prefix + 'test_pre.tsv', sep='\t', header=None)
test_df.head()

# train_df[0] = (train_df[0] == 2).astype(int)
# test_df[0] = (test_df[0] == 2).astype(int)

train_df = pd.DataFrame({
    'id':range(len(train_df)),
    'label':train_df[0],
    'alpha':['a']*train_df.shape[0],
    'text': train_df[1].replace(r'\n', ' ', regex=True)
})

train_df.head()

dev_df = pd.DataFrame({
    'id':range(len(test_df)),
    'label':test_df[0],
    'alpha':['a']*test_df.shape[0],
    'text': test_df[1].replace(r'\n', ' ', regex=True)
})

dev_df.head()

train_df.to_csv('data-adapt/train.tsv', sep='\t', index=False, header=False, columns=train_df.columns)
dev_df.to_csv('data-adapt/dev.tsv', sep='\t', index=False, header=False, columns=dev_df.columns)

# from __future__ import absolute_import, division, print_function

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


from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer,
                                  XLMConfig, XLMForSequenceClassification, XLMTokenizer, 
                                  XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
                                  RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils import (convert_examples_to_features,
                        output_modes, processors)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
    
#     'model_name': '2000-adapt/checkpoint-1700/',

args = {
    'data_dir': 'data-adapt/',
    'model_type':  'bert',
    'model_name': 'bert-base-multilingual-uncased',
    'task_name': 'binary',
    'output_dir': '2000-adapt/checkpoint-1700-adapt/',
    'cache_dir': 'cache/',
    'do_train': True,
    'do_eval': True,
    'do_hold_one_out': False,
    'fp16': False,
    'fp16_opt_level': 'O1',
    'max_seq_length': 512,
    'output_mode': 'classification',
    'train_batch_size': 8,
    'eval_batch_size': 8,
    'gradient_accumulation_steps': 1,
    'num_train_epochs': 8,
    'weight_decay': 0,
    'learning_rate': 4e-5,
    'adam_epsilon': 1e-8,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,

    'logging_steps': 50,
    'evaluate_during_training': True,
    'save_steps': 100,
    'eval_all_checkpoints': False,

    'overwrite_output_dir': False, 
    'reprocess_input_data': False,
    'notes': 'Using Yelp Reviews dataset'
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cuda:0'

args

with open('args.json', 'w') as f:
    json.dump(args, f)

if os.path.exists(args['output_dir']) and os.listdir(args['output_dir']) and args['do_train'] and not args['overwrite_output_dir']:
    raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args['output_dir']))

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}

config_class, model_class, tokenizer_class = MODEL_CLASSES[args['model_type']]

config = config_class.from_pretrained(args['model_name'], num_labels=2, finetuning_task=args['task_name'])
tokenizer = tokenizer_class.from_pretrained(args['model_name'])

model = model_class.from_pretrained(args['model_name'])

# torch.distributed.init_process_group(backend='nccl', world_size=2)
# model = nn.parallel.DistributedDataParallel(model, device_ids=[0,1], output_device=0)

# model = nn.DataParallel(model)
model.to(device)
# model.to('cuda:1')


device

task = args['task_name']

processor = processors[task]()
label_list = processor.get_labels()
num_labels = len(label_list)


def hold_one_out(task, tokenizer, idx, evaluate=False):
    processor = processors[task]()
    output_mode = args['output_mode']
    
    mode = 'dev' if evaluate else 'train'
    
    logger.info("Creating features from dataset file at %s", args['data_dir'])
    label_list = processor.get_labels()
    examples = processor.get_dev_examples(args['data_dir']) if evaluate else processor.get_train_examples(args['data_dir'])
    # For testing
    # examples = examples[0:8]
    if evaluate:
        examples = [examples[idx]]
    else:
        examples.pop(idx)
    print(len(examples))
    # examples  = [example for example in examples if np.random.rand() < undersample_scale_factor]
    examples  = [example for example in examples]
    print(len(examples))
    features = convert_examples_to_features(examples, label_list, args['max_seq_length'], tokenizer, output_mode,
        cls_token_at_end=bool(args['model_type'] in ['xlnet']),            # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        sep_token=tokenizer.sep_token,
        cls_token_segment_id=2 if args['model_type'] in ['xlnet'] else 0,
        pad_on_left=bool(args['model_type'] in ['xlnet']),                 # pad on the left for xlnet
        pad_token_segment_id=4 if args['model_type'] in ['xlnet'] else 0,
        process_count=2)
    
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset

def load_and_cache_examples(task, tokenizer, evaluate=False, undersample_scale_factor=0.01):
    processor = processors[task]()
    output_mode = args['output_mode']
    
    mode = 'dev' if evaluate else 'train'
    cached_features_file = os.path.join(args['data_dir'], f"cached_{mode}_{args['model_name']}_{args['max_seq_length']}_{task}")
    
    if os.path.exists(cached_features_file) and not args['reprocess_input_data']:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
               
    else:
        logger.info("Creating features from dataset file at %s", args['data_dir'])
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(args['data_dir']) if evaluate else processor.get_train_examples(args['data_dir'])
        print(len(examples))
        # examples  = [example for example in examples if np.random.rand() < undersample_scale_factor]
        examples  = [example for example in examples]
        print(len(examples))
        features = convert_examples_to_features(examples, label_list, args['max_seq_length'], tokenizer, output_mode,
            cls_token_at_end=bool(args['model_type'] in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            cls_token_segment_id=2 if args['model_type'] in ['xlnet'] else 0,
            pad_on_left=bool(args['model_type'] in ['xlnet']),                 # pad on the left for xlnet
            pad_token_segment_id=4 if args['model_type'] in ['xlnet'] else 0,
            process_count=2)
        
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
        
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset

def train(train_dataset, model, tokenizer):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args['train_batch_size'])
    
    t_total = len(train_dataloader) // args['gradient_accumulation_steps'] * args['num_train_epochs']
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=args['adam_epsilon'])
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args['warmup_steps'], t_total=t_total)
    
    if args['fp16']:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args['fp16_opt_level'])
        
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args['num_train_epochs'])
    logger.info("  Total train batch size  = %d", args['train_batch_size'])
    logger.info("  Gradient Accumulation steps = %d", args['gradient_accumulation_steps'])
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args['num_train_epochs']), desc="Epoch")
    
    for _ in train_iterator:
        # epoch_iterator = tqdm_notebook(train_dataloader, desc="Iteration")
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      'labels':         batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            print("\r%f" % loss, end='')

            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']

            if args['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args['max_grad_norm'])
                
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])

            tr_loss += loss.item()
            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args['logging_steps'] > 0 and global_step % args['logging_steps'] == 0:
                    # Log metrics
                    if args['evaluate_during_training']:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(model, tokenizer)

                    logging_loss = tr_loss

                if args['save_steps'] > 0 and global_step % args['save_steps'] == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args['output_dir'], 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)


    return global_step, tr_loss / global_step

from sklearn.metrics import mean_squared_error, matthews_corrcoef, confusion_matrix
from scipy.stats import pearsonr

def get_mismatched(labels, preds):
    mismatched = labels != preds
    examples = processor.get_dev_examples(args['data_dir'])
    wrong = [i for (i, v) in zip(examples, mismatched) if v]
    
    return wrong

def get_eval_report(labels, preds):
    mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
   
    acc = ((tn + tp)/(tn + tp + fp + fn))
    prec = ((tp)/(tp + fp))
    rec = ((tp)/(tp + fn))
    f1 = 2 * ((prec * rec) / (prec + rec))
    
    return {
        "mcc": mcc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1
    }, get_mismatched(labels, preds)

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    return get_eval_report(labels, preds)


def evaluate_single(model, tokenizer, eval_dataset, eval_dataloader, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args['output_dir']

    results = {}


    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args['eval_batch_size'])
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    # for batch in tqdm_notebook(eval_dataloader, desc="Evaluating"):
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      'labels':         batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss = outputs[0].detach().cpu().numpy()
            logits = outputs[0]
            pred = logits.argmax().detach().cpu().numpy()
    
    return inputs['labels'].cpu().numpy(), pred, tmp_eval_loss

def evaluate(model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args['output_dir']

    results = {}
    EVAL_TASK = args['task_name']

    eval_dataset = load_and_cache_examples(EVAL_TASK, tokenizer, evaluate=True)
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)


    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args['eval_batch_size'])

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args['eval_batch_size'])
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    # for batch in tqdm_notebook(eval_dataloader, desc="Evaluating"):
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      'labels':         batch[3]}
            outputs = model(**inputs)
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

    if args['output_mode'] == "classification":
        preds = np.argmax(preds, axis=1)
    elif args['output_mode'] == "regression":
        preds = np.squeeze(preds)
    result, wrong = compute_metrics(EVAL_TASK, preds, out_label_ids)
    results.update(result)

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return results, wrong

if args['do_hold_one_out']:

    tns = 0
    fns = 0
    tps = 0
    fps = 0

    right = 0
    wrong = 0

    total = 0

    golds = []
    guesses = []
    losses = []

    failed = 0

    # hack but I'm tired:
    friggin_length = len(hold_one_out(task, tokenizer, 2, evaluate=False)) + 1
    
    EVAL_TASK = args['task_name']

    for i in range(friggin_length):
        try:
            print("############################")
            print("# CURRENTLY ON", i + 1, "of", friggin_length, "#")
            print("############################")
            train_dataset = hold_one_out(task, tokenizer, i, evaluate=False)
            test_dataset = hold_one_out(EVAL_TASK, tokenizer, i, evaluate=True)
       
            test_sampler = SequentialSampler(test_dataset)
            test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args['eval_batch_size'])
            
            
            global_step, tr_loss = train(train_dataset, model, tokenizer)
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

            if not os.path.exists(args['output_dir']):
                    os.makedirs(args['output_dir'])
            logger.info("Saving model checkpoint to %s", args['output_dir'])
            
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args['output_dir'])
            tokenizer.save_pretrained(args['output_dir'])
            torch.save(args, os.path.join(args['output_dir'], 'training_args.bin'))

            # OLD EVAL
            checkpoints = [args['output_dir']]
            if args['eval_all_checkpoints']:
                checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args['output_dir'] + '/**/' + WEIGHTS_NAME, recursive=True)))
                logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            logger.info("Evaluate the following checkpoints: %s", checkpoints)
            checkpoint = checkpoints[-1]
            # for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(device)
            # SINGLE EVAL
            gold, guess, loss = evaluate_single(model, tokenizer, test_dataset, test_dataloader, prefix=global_step)
            
            gold = int(gold)
            guess = int(guess)
            
            golds.append(gold)
            guesses.append(guess)
            losses.append(float(loss))

            total += 1

            print('\t'.join(["ANSWER", str(gold), str(guess)]))

            if gold == guess:
                right += 1
                if gold == 1:
                    tps += 1
                else:
                    tns += 1
            else:
                wrong += 1
                if gold == 0:
                    fps += 1
                else:
                    fns += 1


            print("gold", gold)
            print("guess", guess)
            print("tps:", tps)
            print("tns:", tns)
            print("fps:", fps)
            print("fns:", fns)
            

        except:
            failed += 1

        def divide(numer, denom):
            if denom != 0 and type(denom) == int:
                return numer / denom
            elif denom == 0:
                return 0.0
            else:
                return 99999.999999

        acc = divide(right, total)
        prec = divide(tps, tps + fps)
        rec = divide(tps, tps + fns)
        f1 = 2 * divide(prec * rec, prec + rec) 

        print("Accuracy:", acc, right, total)
        print("Precision:", prec, tps, tps + fps)
        print("Recall:", rec, tps, tps + fns)
        print("F1:", f1)
        print("failed", failed)

else:
    # IMPORTANT #
    # Due to the 12 hour limit on Google Colab and the time it would take to convert the dataset into features, the load_and_cache_examples() function has been modified
    # to randomly undersample the dataset by a scale of 0.1
    # AAAAAAAAAAHHHHHHHHHHHHH THIS KILLED ME --- DLK

    if args['do_train']:
        train_dataset = load_and_cache_examples(task, tokenizer, undersample_scale_factor=0.1)
        global_step, tr_loss = train(train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args['do_train']:
        if not os.path.exists(args['output_dir']):
                os.makedirs(args['output_dir'])
        logger.info("Saving model checkpoint to %s", args['output_dir'])
        
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args['output_dir'])
        tokenizer.save_pretrained(args['output_dir'])
        torch.save(args, os.path.join(args['output_dir'], 'training_args.bin'))

    results = {}
    if args['do_eval']:
        checkpoints = [args['output_dir']]
        if args['eval_all_checkpoints']:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args['output_dir'] + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        checkpoint = checkpoints[-1]
        # for checkpoint in checkpoints:
        global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
        model = model_class.from_pretrained(checkpoint)
        model.to(device)
        # Original eval
        result, wrong_preds = evaluate(model, tokenizer, prefix=global_step)
        result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
        results.update(result)

    results

