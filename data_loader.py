# -*- coding: utf-8 -*-
import sys
import torch
import torch.autograd as autograd
import codecs
import random
import torch.utils.data as Data

SEED = 1

# input: a sequence of tokens, and a token_to_index dictionary
# output: a LongTensor variable to encode the sequence of idxs
def prepare_sequence(seq, to_ix, cuda=False):
    var = autograd.Variable(torch.LongTensor([to_ix[w] for w in seq.split(' ')]))
    return var

def prepare_label(label,label_to_ix, cuda=False):
    var = autograd.Variable(torch.LongTensor([label_to_ix[label]]))
    return var

def build_token_to_ix(sentences):
    token_to_ix = dict()
    print(len(sentences))
    for sent in sentences:
        for token in sent.split(' '):
            if token not in token_to_ix:
                token_to_ix[token] = len(token_to_ix)
    token_to_ix['<pad>'] = len(token_to_ix)
    return token_to_ix

def build_label_to_ix(labels):
    label_to_ix = dict()
    for label in labels:
        if label not in label_to_ix:
            label_to_ix[label] = len(label_to_ix)


def load_MR_data():

    # already tokenized and there is no standard split
    # the size follow the Mou et al. 2016 instead
    file_pos = './datasets/MR/rt-polarity.pos'
    file_neg = './datasets/MR/rt-polarity.neg'
    print('loading MR data from',file_pos,'and',file_neg)

    pos_sents = codecs.open(file_pos, 'r', 'utf8').read().split('\n')
    neg_sents = codecs.open(file_neg, 'r', 'utf8').read().split('\n')

    random.seed(SEED)
    random.shuffle(pos_sents)
    random.shuffle(neg_sents)

    # print(len(pos_sents))
    # print(len(neg_sents))

    train_data = [(sent,1) for sent in pos_sents[:4250]] + [(sent, 0) for sent in neg_sents[:4250]]
    dev_data = [(sent, 1) for sent in pos_sents[4250:4800]] + [(sent, 0) for sent in neg_sents[4250:4800]]
    test_data = [(sent, 1) for sent in pos_sents[4800:]] + [(sent, 0) for sent in neg_sents[4800:]]


    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)

    print('train:',len(train_data),'dev:',len(dev_data),'test:',len(test_data))

    word_to_ix = build_token_to_ix([s for s,_ in train_data+dev_data+test_data])
    label_to_ix = {0:0,1:1}
    print('vocab size:',len(word_to_ix),'label size:',len(label_to_ix))
    print('loading data done!')
    return train_data,dev_data,test_data,word_to_ix,label_to_ix


def load_MR_data_batch():

    pass