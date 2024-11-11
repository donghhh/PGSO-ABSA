



# This file contains all data loading and transformation functions
import random

import numpy as np
import spacy
import pickle
import time

import torch
from torch.utils.data import Dataset
from spacy.tokens import Doc
class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}

dep2id = {'ROOT': 1, 'acl': 2, 'acomp': 3, 'advcl': 4, 'advmod': 5, 'agent': 6, 'amod': 7, 'appos': 8,
          'attr': 9, 'auxpass': 10, 'case': 11, 'cc': 12, 'ccomp': 13, 'compound': 14, 'conj': 15,
          'csubj': 16, 'csubjpass': 17, 'dative': 18, 'dep': 19, 'det': 20, 'dobj': 21, 'expl': 22,
          'intj': 23, 'mark': 24, 'meta': 25, 'neg': 26, 'nmod': 27, 'npadvmod': 28, 'nsubj': 29,
          'nsubjpass': 30, 'nummod': 31, 'oprd': 32, 'parataxis': 33, 'pcomp': 34, 'pobj': 35, 'poss': 36,
          'preconj': 37, 'predet': 38, 'prep': 39, 'prt': 40, 'punct': 41, 'quantmod': 42, 'relcl': 43, 'xcom': 44,
          'aux': 45, 'xcomp': 46, 'self': 47}
pos2id = {'<pad>': 0, 'NN': 1, 'DT': 2, 'JJ': 3, 'IN': 4, 'RB': 5, '.': 6, 'CC': 7, ',': 8, 'PRP': 9,
                   'NNS': 10, 'VBD': 11, 'VB': 12, 'VBZ': 13, 'NNP': 14, 'VBP': 15, 'VBN': 16, 'PRP$': 17, 'VBG': 18,
                   'TO': 19, 'CD': 20, 'MD': 21, 'HYPH': 22, '-LRB-': 23, '-RRB-': 24, ':': 25, 'WDT': 26, 'RP': 27,
                   'JJS': 28, 'WRB': 29, 'JJR': 30, 'WP': 31, 'POS': 32, '$': 33, 'EX': 34, 'PDT': 35,
                   'RBR': 36, 'UH': 37, 'RBS': 38, 'SYM': 39, 'NNPS': 40, 'NFP': 41, 'FW': 42,
                   '``': 43, "''": 44, 'XX': 45, 'LS':46,'AFX':47, 'ADD':48, '_SP':49}
length = len(dep2id)
def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    tokens = nlp(text)
    words = text.split()
    #pos=[]
    #matrix = np.zeros((len(words), len(words))).astype('float32')
    matrix = np.zeros((128,128)).astype('float32')
    matrix1 = np.zeros((128, 128)).astype('float32')
    pos = np.zeros((128)).astype('float32')
    root_list=[]
    assert len(words) == len(list(tokens))
    for token in tokens:
        if token.i < len(words)-1:  # not include "."
            matrix[token.i][token.i] = int(dep2id['self'])
            if token.tag_ not in pos2id.keys():
                pos2id.update({token.tag_:max(pos2id.values())+1})
                print(token.tag_)
            pos[token.i] = pos2id[token.tag_]
            dep_tag = token.dep_
            if dep_tag not in dep2id.keys():
                dep2id.update({dep_tag:max(dep2id.values())+1})
                print(dep_tag)
            head_token = token.head
            if dep_tag == 'ROOT':
                root_list.append(token.i)

            matrix1[token.i][head_token.i] = dep2id[dep_tag]
            matrix[head_token.i][token.i] = dep2id[dep_tag]
    return matrix1,pos

def read_line_examples_from_file(data_path):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, labels, id2matrix,poses= [], [] ,{},[]
    key=0
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            if line != '':
                words, tuples = line.split('####')
                matrix,pos = dependency_adj_matrix(words)
                poses.append(pos)
                sents.append(words.split())
                labels.append(eval(tuples))
                id2matrix[key] = matrix
                key+=1
    print(f"Total examples = {len(sents)}")
    return sents, labels,id2matrix,poses

def get_extraction_uabsa_targets(sents, labels):
    targets = []
    for i, label in enumerate(labels):
        if label == []:
            targets.append('None')
        else:
            all_tri = []
            for tri in label:
                if len(tri[0]) == 1:
                    a = sents[i][tri[0][0]]
                else:
                    start_idx, end_idx = tri[0][0], tri[0][-1]
                    a = ' '.join(sents[i][start_idx:end_idx+1])
                c = senttag2word[tri[1]]
                all_tri.append((a, c))
            label_strs = ['<extra_id_0> ' + l[0] + ' <extra_id_1> ' + l[1] for l in all_tri]
            targets.append(' <extra_id_2> '.join(label_strs))
            # label_strs = ['('+', '.join(l)+')' for l in all_tri]
            # targets.append('; '.join(label_strs))
    return targets


def get_extraction_aope_targets(sents, labels):
    targets = []
    for i, label in enumerate(labels):
        all_tri = []
        for tri in label:
            if len(tri[0]) == 1:
                a = sents[i][tri[0][0]]
            else:
                start_idx, end_idx = tri[0][0], tri[0][-1]
                a = ' '.join(sents[i][start_idx:end_idx+1])
            if len(tri[1]) == 1:
                b = sents[i][tri[1][0]]
            else:
                start_idx, end_idx = tri[1][0], tri[1][-1]
                b = ' '.join(sents[i][start_idx:end_idx+1])
            all_tri.append((a, b))
        label_strs = ['<extra_id_0> ' + l[0] + ' <extra_id_1> ' + l[1] for l in all_tri]
        targets.append(' <extra_id_2> '.join(label_strs))
    return targets


def get_extraction_tasd_targets(sents, labels):
    targets = []
    for label in labels:
        label_strs = ['<extra_id_0> '+ l[0] + ' <extra_id_1> ' + l[1] + ' <extra_id_2> '+l[2] for l in label]
        # targets.append(' '.join(label_strs))
        targets.append(' <extra_id_3> '.join(label_strs))
        # targets.append(target)
    # targets = []
    # for label in labels:
    #     label_strs = ['(' + ', '.join(l) + ')' for l in label]
    #     target = '; '.join(label_strs)
    #     targets.append(target)
    return targets

def get_extraction_acos_targets(sents, labels):
    targets = []
    for label in labels:
        label_strs = ['<extra_id_0> '+ l[0] + ' <extra_id_1> ' + l[1].lower().replace('#',' ').replace('_',' ') + ' <extra_id_2> '+l[2] + ' <extra_id_3> ' + l[3] for l in label]
        # targets.append(' '.join(label_strs))
        targets.append(' <extra_id_4> '.join(label_strs))
        # targets.append(target)
    # targets = []
    # for label in labels:
    #     label_strs = ['(' + ', '.join(l) + ')' for l in label]
    #     target = '; '.join(label_strs)
    #     targets.append(target)
    return targets



def get_extraction_aste_targets(sents, labels):
    targets = []

    tuple = 0
    for i, label in enumerate(labels):
        # distance = 0
        all_tri = []
        for tri in label:

            if len(tri[0]) == 1:
                a = sents[i][tri[0][0]]
                aspect = tri[0][0]
            else:
                start_idx, end_idx = tri[0][0], tri[0][-1]
                aspect = int((tri[0][0]+tri[0][1])/2)
                a = ' '.join(sents[i][start_idx:end_idx+1])
            if len(tri[1]) == 1:
                b = sents[i][tri[1][0]]
                opinion = tri[1][0]
            else:
                start_idx, end_idx = tri[1][0], tri[1][-1]
                opinion = int((tri[1][0]+tri[1][1])/2)
                b = ' '.join(sents[i][start_idx:end_idx+1])
            c = senttag2word[tri[2]]
            #c = senttag2wordl[tri[2]]
            all_tri.append((a, b, c))
            #all_tri.append((a+ b, c))
        #     distance = distance + abs(opinion-aspect)
        # if distance >7:
        #     tuple = tuple+1
        label_strs = ['<extra_id_0> '+l[0] + ' <extra_id_1> ' + l[1]+' <extra_id_2> '+l[2] for l in all_tri]
        targets.append(' <extra_id_3> '.join(label_strs))

    return targets


def get_transformed_io(data_path, paradigm, task):
    """
    The main function to transform the Input & Output according to 
    the specified paradigm and task
    """
    sents, labels,idx2matrix,tags = read_line_examples_from_file(data_path)

    inputs = [s.copy() for s in sents]

    # directly treat label infor as the target
    if task == 'uabsa':
        targets = get_extraction_uabsa_targets(sents, labels)
    elif task == 'aste':
        targets = get_extraction_aste_targets(sents, labels)
    elif task == 'tasd':
        targets = get_extraction_tasd_targets(sents, labels)
    elif task == 'aope':
        targets = get_extraction_aope_targets(sents, labels)
    elif task =='acos':
        targets = get_extraction_acos_targets(sents,labels)
    else:
        raise NotImplementedError

    return inputs, targets, idx2matrix, tags


class ABSADataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type, paradigm, task, max_len=128):
        # 'data/aste/rest16/train.txt'
        self.data_path = f'data/{task}/{data_dir}/{data_type}.txt'
        self.paradigm = paradigm
        self.task = task
        self.max_len = max_len
        self.tokenizer = tokenizer

        self.inputs = []
        self.targets = []
        self.matrixs=[]
        self.tags = []
        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        matrix = self.matrixs[index]
        tag = self.tags[index]

        src_mask = self.inputs[index]["attention_mask"].squeeze()      # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        offset_mapping  = self.inputs[index]["offset_mapping"].squeeze()
        return {"source_ids": source_ids, "source_mask": src_mask, 
                "target_ids": target_ids, "target_mask": target_mask,
                "matrix":matrix,"tag":tag,"offset_mapping":offset_mapping}

    def _build_examples(self):

        inputs, targets,idx2matrix,tags = get_transformed_io(self.data_path, self.paradigm, self.task)
        for i in range(len(inputs)):
            matrix = idx2matrix[i]
            tag = tags[i]
            input = ' '.join(inputs[i]) 
            if self.paradigm == 'annotation':
                if self.task != 'tasd':
                    target = ' '.join(targets[i]) 
                else:
                    target = targets[i]
            else:
                target = targets[i]

            tokenized_input = self.tokenizer.batch_encode_plus(
              [input], max_length=self.max_len, pad_to_max_length=True, truncation=True,
              return_tensors="pt",return_offsets_mapping=True
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
              [target], max_length=self.max_len, pad_to_max_length=True, truncation=True,
              return_tensors="pt"
            )

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)
            self.matrixs.append(matrix)
            self.tags.append(tag)


def write_results_to_log(log_file_path, best_test_result, args, dev_results, test_results, global_steps):
    """
    Record dev and test results to log file
    """
    local_time = time.asctime(time.localtime(time.time()))
    exp_settings = "Exp setting: {0} on {1} under {2} | {3:.4f} | ".format(
        args.task, args.dataset, args.paradigm, best_test_result
    )
    train_settings = "Train setting: bs={0}, lr={1}, num_epochs={2}".format(
        args.train_batch_size, args.learning_rate, args.num_train_epochs
    )
    results_str = "\n* Results *:  Dev  /  Test  \n"

    metric_names = ['f1', 'precision', 'recall']
    for gstep in global_steps:
        results_str += f"Step-{gstep}:\n"
        for name in metric_names:
            name_step = f'{name}_{gstep}'
            results_str += f"{name:<8}: {dev_results[name_step]:.4f} / {test_results[name_step]:.4f}"
            results_str += ' '*5
        results_str += '\n'

    log_str = f"{local_time}\n{exp_settings}\n{train_settings}\n{results_str}\n\n"

    with open(log_file_path, "a+") as f:
        f.write(log_str)