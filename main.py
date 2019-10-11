import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from sklearn.metrics import f1_score

from collections import OrderedDict
from tqdm import tqdm_notebook as tqdm 
import math, copy, time
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context(context="talk")

import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd

from data_loader import loader 
from data_loader_map import loader_map
from trainer import trainer 
from models import RNNModel, CNNmodel, CNN, RNN, RNN_CNNmodel

torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

src = "./HOT_cleaned.csv"
emb_src = "./glove.6B.50d.txt"
train_data, val_data, test_data, vocab, weights_matrix = loader(src, emb_src, 50 , 50)

vocab_size = len(vocab)

output_dim  =  3
emb_dim = 50 
hid_dim = 100 

model = RNNModel(output_dim, emb_dim , hid_dim , weights_matrix,dropout = 0.5).cuda()
optimizer = optim.Adam(model.parameters(), lr= 1e-3)
criterion = nn.BCEWithLogitsLoss().cuda()

trainer(model, train_data, val_data,test_data, optimizer, criterion, 20)

embedding_dim = 50
n_filters = 100
filter_sizes = [3,4,5]
output_dim = 3
dropout = 0.5

model = CNNmodel(embedding_dim, n_filters , filter_sizes, output_dim, weights_matrix ,dropout).cuda()
optimizer = optim.Adam(model.parameters(), lr= 1e-2)
criterion = nn.BCEWithLogitsLoss().cuda()

trainer(model, train_data, val_data,test_data, optimizer, criterion, 20)

embedding_dim = 50
n_filters = 100
filter_sizes = [3,4,5]
output_dim = 3
dropout = 0.3

cnn = CNN(embedding_dim, n_filters , filter_sizes, output_dim,weights_matrix, dropout).cuda()
rnn = RNN(vocab_size, 50, 100, weights_matrix, dropout = dropout).cuda()

model = RNN_CNNmodel(cnn, rnn, 3, dropout).cuda()
optimizer = optim.Adam(model.parameters(), lr= 1e-3)
criterion = nn.BCEWithLogitsLoss().cuda()

trainer(model, train_data, val_data,test_data, optimizer, criterion, 20)

src = "./HOT_cleaned.csv"
emb_src = "./glove.6B.50d.txt"
prof_src = "./Hinglish_Profanity_List.csv"
train_data, val_data, test_data, vocab, weights_matrix = loader_map(src, emb_src, prof_src, 50 , 50)

output_dim  =  3
emb_dim = 50 
hid_dim = 100 

model = RNNModel(output_dim, emb_dim , hid_dim , weights_matrix,dropout = 0.5).cuda()
optimizer = optim.Adam(model.parameters(), lr= 1e-3)
criterion = nn.BCEWithLogitsLoss().cuda()

trainer(model, train_data, val_data,test_data, optimizer, criterion, 20)

embedding_dim = 50
n_filters = 100
filter_sizes = [3,4,5]
output_dim = 3
dropout = 0.5

model = CNNmodel(embedding_dim, n_filters , filter_sizes, output_dim, weights_matrix ,dropout).cuda()
optimizer = optim.Adam(model.parameters(), lr= 1e-2)
criterion = nn.BCEWithLogitsLoss().cuda()

trainer(model, train_data, val_data,test_data, optimizer, criterion, 20)

embedding_dim = 50
n_filters = 100
filter_sizes = [3,4,5]
output_dim = 3
dropout = 0.3

cnn = CNN(embedding_dim, n_filters , filter_sizes, output_dim,weights_matrix, dropout).cuda()
rnn = RNN(vocab_size, 50, 100, weights_matrix, dropout = dropout).cuda()

model = RNN_CNNmodel(cnn, rnn, 3, dropout).cuda()
optimizer = optim.Adam(model.parameters(), lr= 1e-2)
criterion = nn.BCEWithLogitsLoss().cuda()

trainer(model, train_data, val_data,test_data, optimizer, criterion, 20)
