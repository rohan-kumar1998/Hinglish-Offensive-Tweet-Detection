import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from sklearn.metrics import f1_score

from tqdm import tqdm 
import math, copy, time
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")

import numpy as np

class RNNModel(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, weights_matrix, dropout = 0.5):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(weights_matrix, freeze = False)
        self.rnn = nn.GRU(emb_dim, hid_dim, bidirectional = True)
        self.fc = nn.Linear(2*hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        src = src.permute(1,0)
        embedded = self.embedding(src)
        output, hidden = self.rnn(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        out = self.fc(hidden.squeeze())
        return out


class CNNmodel(nn.Module):
    def __init__(self, embedding_dim, n_filters, filter_sizes, output_dim, weights_matrix, dropout):
        super(CNNmodel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(weights_matrix, freeze = False)
        self.conv_0 = nn.Conv1d(in_channels = 1, out_channels = n_filters, kernel_size=(filter_sizes[0], embedding_dim))
        self.conv_1 = nn.Conv1d(in_channels = 1, out_channels = n_filters, kernel_size=(filter_sizes[1], embedding_dim))
        self.conv_2 = nn.Conv1d(in_channels = 1, out_channels = n_filters, kernel_size=(filter_sizes[2], embedding_dim))
        self.linear = nn.Linear(len(filter_sizes)*n_filters, output_dim)#
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        embedded = self.embedding(x) #o/p->[batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)#o/p->[batch size,1, sent len, emb dim]
        conved_0 = F.relu(self.conv_0(embedded.squeeze(3)))
        conved_1 = F.relu(self.conv_1(embedded.squeeze(3)))
        conved_2 = F.relu(self.conv_2(embedded.squeeze(3)))
        pooled_0 = F.max_pool1d(conved_0.squeeze(3), conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1.squeeze(3), conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2.squeeze(3), conved_2.shape[2]).squeeze(2)
        out = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim = 1))
        out = self.linear(out)
        return out 


class RNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, weights_matrix, dropout = 0.5):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(weights_matrix, freeze = False)
        self.rnn = nn.GRU(emb_dim, hid_dim, bidirectional = True)
        self.fc = nn.Linear(2*hid_dim, 3)
        self.hid_dim = 2*hid_dim 
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        src = src.permute(1,0)
        embedded = self.embedding(src)
        output, hidden = self.rnn(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return hidden

class CNN(nn.Module):
    def __init__(self, embedding_dim, n_filters, filter_sizes, output_dim, weights_matrix, dropout):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(weights_matrix, freeze = False)
        self.conv_0 = nn.Conv1d(in_channels = 1, out_channels = n_filters, kernel_size=(filter_sizes[0], embedding_dim))
        self.conv_1 = nn.Conv1d(in_channels = 1, out_channels = n_filters, kernel_size=(filter_sizes[1], embedding_dim))
        self.conv_2 = nn.Conv1d(in_channels = 1, out_channels = n_filters, kernel_size=(filter_sizes[2], embedding_dim))
        self.linear = nn.Linear(len(filter_sizes)*n_filters, output_dim)#
        self.len_layer = len(filter_sizes)*n_filters
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        embedded = self.embedding(x) #o/p->[batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)#o/p->[batch size,1, sent len, emb dim]
        conved_0 = F.relu(self.conv_0(embedded.squeeze(3)))
        conved_1 = F.relu(self.conv_1(embedded.squeeze(3)))
        conved_2 = F.relu(self.conv_2(embedded.squeeze(3)))
        pooled_0 = F.max_pool1d(conved_0.squeeze(3), conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1.squeeze(3), conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2.squeeze(3), conved_2.shape[2]).squeeze(2)
        out = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim = 1))
        return out 

class RNN_CNNmodel(nn.Module):
    def __init__(self, cnn, rnn, output_dim, dropout):
        super().__init__()
        self.cnn = cnn
        self.rnn = rnn 
        self.fc1 = nn.Linear(cnn.len_layer + rnn.hid_dim, 100)
        self.maxpool = nn.MaxPool1d(2)
        self.fc2 = nn.Linear(50, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out1 = self.cnn(x)
        out2 = self.rnn(x)
        out = torch.cat((out1,out2), dim = 1)
        out = self.fc1(out)
        out = out.unsqueeze(0)
        out = self.maxpool(out)
        out = out.squeeze()
        out = self.fc2(out)
        return out
    