# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_loader
import os
import random
torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)

class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y  = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y)
        return log_probs



def get_accuracy(truth, pred):
     assert len(truth)==len(pred)
     right = 0
     for i in range(len(truth)):
         if truth[i]==pred[i]:
             right += 1.0
     return right/len(truth)

def train():
    train_data, dev_data, test_data, word_to_ix, label_to_ix = data_loader.load_MR_data()
    EMBEDDING_DIM = 50
    HIDDEN_DIM = 50
    EPOCH = 100
    best_dev_acc = 0.0
    model = LSTMClassifier(embedding_dim=EMBEDDING_DIM,hidden_dim=HIDDEN_DIM,
                           vocab_size=len(word_to_ix),label_size=len(label_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(),lr = 1e-3)
    #optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2)
    no_up = 0
    for i in range(EPOCH):
        random.shuffle(train_data)
        print('epoch: %d start!' % i)
        train_epoch(model, train_data, loss_function, optimizer, word_to_ix, label_to_ix, i)
        print('now best dev acc:',best_dev_acc)
        dev_acc = evaluate(model,dev_data,loss_function,word_to_ix,label_to_ix,'dev')
        test_acc = evaluate(model, test_data, loss_function, word_to_ix, label_to_ix, 'test')
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            os.system('rm mr_best_model_acc_*.model')
            print('New Best Dev!!!')
            torch.save(model.state_dict(), 'best_models/mr_best_model_acc_' + str(int(test_acc*10000)) + '.model')
            no_up = 0
        else:
            no_up += 1
            if no_up >= 10:
                exit()

def evaluate(model, data, loss_function, word_to_ix, label_to_ix, name ='dev'):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []

    for sent, label in data:
        truth_res.append(label_to_ix[label])
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()
        sent = data_loader.prepare_sequence(sent, word_to_ix)
        label = data_loader.prepare_label(label, label_to_ix)
        pred = model(sent)
        pred_label = pred.data.max(1)[1].numpy()
        pred_res.append(pred_label)
        # model.zero_grad() # should I keep this when I am evaluating the model?
        loss = loss_function(pred, label)
        avg_loss += loss.data[0]
    avg_loss /= len(data)
    acc = get_accuracy(truth_res, pred_res)
    print(name + ' avg_loss:%g train acc:%g' % (avg_loss, acc ))
    return acc



def train_epoch(model, train_data, loss_function, optimizer, word_to_ix, label_to_ix, i):
    model.train()
    
    avg_loss = 0.0
    count = 0
    truth_res = []
    pred_res = []
    batch_sent = []

    for sent, label in train_data:


        truth_res.append(label_to_ix[label])
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()
        sent = data_loader.prepare_sequence(sent, word_to_ix)
        label = data_loader.prepare_label(label, label_to_ix)
        pred = model(sent)
        pred_label = pred.data.max(1)[1].numpy()
        pred_res.append(pred_label)
        model.zero_grad()
        loss = loss_function(pred, label)
        avg_loss += loss.data[0]
        count += 1
        if count % 500 == 0:
            print('epoch: %d iterations: %d loss :%g' % (i, count, loss.data[0]))

        loss.backward()
        optimizer.step()
    avg_loss /= len(train_data)
    print('epoch: %d done! \n train avg_loss:%g , acc:%g'%(i, avg_loss, get_accuracy(truth_res,pred_res)))

train()

