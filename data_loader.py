import numpy as np
import pandas as pd
import nltk
import os
from tqdm import tqdm 
from nltk.tokenize import RegexpTokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math, copy, time
import pdb
from torch.utils.data import Dataset, DataLoader
from torchtext import data, datasets 
from torch.nn.utils.rnn import pad_sequence

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.model_selection import train_test_split

class dataset(Dataset):
    def __init__(self, df, emb_src ,emb_dim ,vocab = None):
        super(dataset, self).__init__()
        if(vocab == None):
            self.vocab = self.buid_vocab(df)
            self.weights_matrix = self.align_pretrained(self.vocab, emb_src, emb_dim)
        else:
            self.vocab = vocab
        data = []
        for i in range(len(df)):
            if(int(df['score'][i]) == 0):
                data.append({'untext' : df['text'][i],'text' : torch.LongTensor(self.preprocess(df['text'][i], self.vocab)), 'score' : df['score'][i],  'label' : torch.Tensor([1,0,0])})    
            elif(int(df['score'][i]) == 1):
                data.append({'untext' : df['text'][i],'text' : torch.LongTensor(self.preprocess(df['text'][i], self.vocab)), 'score' : df['score'][i],  'label' : torch.Tensor([0,1,0])})
            else:
                data.append({'untext' : df['text'][i],'text' : torch.LongTensor(self.preprocess(df['text'][i], self.vocab)), 'score' : df['score'][i],  'label' : torch.Tensor([0,0,1])})    
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def preprocess(self,sentence, vocab):
        sentence = sentence.lower()
        tokenizer = RegexpTokenizer("\w+\'?\w+|\w+")
        tokens = tokenizer.tokenize(sentence)
        word2ids = [vocab['<sos>']]
        word2ids.extend([vocab[word] for word in tokens if word in vocab.keys()])
        word2ids.append(vocab['<eos>'])
        return word2ids
    
    def buid_vocab(self,df):
        words = []
        tokenizer = RegexpTokenizer("\w+\'?\w+|\w+")
        for i in range(len(df)):
            sentence = df['text'][i]
            words.extend(tokenizer.tokenize(sentence))
#         print(len(words))
        words = list(set(words))
#         print(len(words))
        vocab = {}
        vocab['<pad>'] = 0
        vocab['<sos>'] = 1  
        vocab['<eos>'] = 2  
        idx = 3 
        for word in words:
            vocab[word] = idx
            idx += 1 
        return vocab    
    
    def align_pretrained(self, vocab, emb_src, emb_dim):
        glove_file = datapath(emb_src)
        tmp_file = get_tmpfile("test_word2vec.txt")

        _ = glove2word2vec(glove_file, tmp_file)

        model = KeyedVectors.load_word2vec_format(tmp_file)
        
        matrix_len = len(vocab) + 1
        weights_matrix = np.zeros((matrix_len, emb_dim))       
        for _,k in enumerate(vocab):
            try:
#                 print(vocab[k], " ", k)
                weights_matrix[vocab[k]] = model[k]
            except:
                weights_matrix[vocab[k]] = np.random.normal(scale=0.6, size=(emb_dim, ))
        return torch.FloatTensor(weights_matrix)

def collater(batch):
    sequences = [item['text'] for item in batch]
    untext = [item['untext'] for item in batch]
    score = torch.Tensor([item['score'] for item in batch])
    label = [item['label'] for item in batch]
    label = pad_sequence(label, batch_first= True, padding_value= 0)
    text = pad_sequence(sequences, batch_first=True, padding_value=0)
    new_batch = {'text' : text, 'untext' : untext, 'score' : score, 'label' : label}
    return new_batch

def data_split(df):    
    train, test = train_test_split(df, test_size=0.4)
    test, val = train_test_split(test, test_size = 0.25)
    return train.reset_index(), val.reset_index(), test.reset_index()

def loader(src,emb_src, emb_dim, batch_size): #add emb_src, emb_dim
    """
    Returns Training iterator, Testing iterator, vocab size and weights matrix.  
    """
    df = pd.read_csv( src, index_col=None, header=0, engine='python' )
    train_df, val_df, test_df = data_split(df)
    train_data = dataset(train_df, emb_src, emb_dim) #add emb_src, emb_dim
    vocab = train_data.vocab
    weights_matrix = train_data.weights_matrix
    val_data = dataset(val_df, emb_src, emb_dim, vocab)
    test_data = dataset(test_df, emb_src, emb_dim, vocab)
    train_data = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, collate_fn= collater ,drop_last=True)
    test_data = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, collate_fn= collater ,drop_last=True)
    val_data = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True, collate_fn= collater ,drop_last=True)
    return train_data, val_data, test_data, vocab, weights_matrix