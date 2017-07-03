# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_loader
torch.manual_seed(1)

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



def train():
    training_data, word_to_ix, label_to_ix = data_loader.load_data()
    print(training_data)
    print(word_to_ix)
    print(label_to_ix)

    EMBEDDING_DIM = 6
    HIDDEN_DIM = 6
    EPOCH = 5

    model = LSTMClassifier(embedding_dim=EMBEDDING_DIM,hidden_dim=HIDDEN_DIM,
                           vocab_size=len(word_to_ix),label_size=len(label_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(),lr = 1e-3)

    for i in range(EPOCH):
        avg_loss = 0.0
        count = 0
        for sent, label in training_data:

            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()
            sent = data_loader.prepare_sequence(sent, word_to_ix)
            label = data_loader.prepare_label(label, label_to_ix)
            pred = model(sent)
            model.zero_grad()
            loss = loss_function(pred, label)
            avg_loss += loss.data[0]

            count+=1
            print('epoch :%d iterations :%d loss :%g' % (i, count, loss.data[0]))

            loss.backward()
            optimizer.step()
        # torch.save(model.state_dict(), 'model' + str(i + 1) + '.pth')
        print('the average loss after completion of %d epochs is %g'
              % ((i + 1), (avg_loss / len(training_data))))


train()
