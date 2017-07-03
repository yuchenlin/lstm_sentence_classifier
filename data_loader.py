# -*- coding: utf-8 -*-
import sys
import torch
import torch.autograd as autograd


# input: a sequence of tokens, and a token_to_index dictionary
# output: a LongTensor variable to encode the sequence of idxs
def prepare_sequence(seq, to_ix):
    return autograd.Variable(torch.LongTensor([to_ix[w] for w in seq]))

def prepare_label(label,label_to_ix):
    return autograd.Variable(torch.LongTensor([label_to_ix[label]]))

def build_token_to_ix(sentences):
    token_to_ix = dict()
    for sent in sentences:
        for token in sent:
            if token not in token_to_ix:
                token_to_ix[token] = len(token_to_ix)

    return token_to_ix

def build_label_to_ix(labels):
    label_to_ix = dict()
    for label in labels:
        if label not in label_to_ix:
            label_to_ix[label] = len(label_to_ix)

def load_data():
    training_data = [("The dog ate the apple".split(), 0),
                    ("Everybody read that book".split(), 1)]

    word_to_ix = build_token_to_ix([sen for sen,_ in training_data])

    label_to_ix = {0:0,1:1}

    return training_data, word_to_ix , label_to_ix
