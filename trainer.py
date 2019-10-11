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

def evaluate(model, test_data, test = 0):
  model.eval()
  denom = 0.0
  numer = 0.0
  y_true = [] 
  y_pred = []
  with torch.no_grad():
    for batch in test_data:
      trg = Variable(batch['label']).cuda()
      src = Variable(batch['text']).cuda()
      out = model(src)
      out = out.squeeze() 
      out = out.data 
      trg = trg.data 
      _,trg1 = torch.max(trg, dim = 1)
      _,out1 = torch.max(out, dim = 1)
      trg1 = trg1.data.cpu().numpy() 
      out1 = out1.data.cpu().numpy() 
      denom += len(trg1)
      for i in range(len(trg1)):
        if(trg1[i] == out1[i]):
          numer += 1
      y_true.extend(trg1.tolist())
      y_pred.extend(out1.tolist())
    
    if(test):
      labels = [0,1,2]
      conf = confusion_matrix(y_true, y_pred, labels = labels)
      df_cm = pd.DataFrame(conf, index = labels, columns = labels)
      plt.figure(figsize = (10,7))
      plt.title('Confusion Matrix')
      plt.xlabel('Predicted')
      plt.ylabel('Actual')
      sns.heatmap(df_cm.astype(int) , annot=True,cmap="Blues",fmt='g')
      
  return (float(numer)/float(denom))*100

def trainer(model,train_data,val_data,test_data, optimizer, criterion, epochs):
  running_loss = []
  val_loss = []
  running_epochs  = []
  train_accuracy = []
  val_accuracy = []
  best_loss = math.inf
  best_model_state_dict = {k:v for k, v in model.state_dict().items()}
  best_model_state_dict = OrderedDict(best_model_state_dict)
  for epoch in tqdm(range(epochs)):  
    epoch_loss = 0 
    epoch_acc = 0
    epoch_size = 0
    
    model.train()
    
    for batch in tqdm(train_data):
        optimizer.zero_grad()
        src = Variable(batch['text']).cuda()
        out = model(src)
        out = out.squeeze()
        trg = Variable(batch['label']).cuda()
        loss = criterion(out, trg)
        loss.backward()    
        clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
    
    model.eval()
    
    train_accuracy.append(evaluate(model, train_data))
    running_loss.append(epoch_loss/len(train_data))
    running_epochs.append(epoch+1)
    epoch_loss = 0
    
    with torch.no_grad():
      for batch in tqdm(val_data):
          src = Variable(batch['text']).cuda()
          out = model(src)
          out = out.squeeze()
          trg = Variable(batch['label']).cuda()
          loss = criterion(out, trg)
          epoch_loss += loss.item()

    val_accuracy.append(evaluate(model, val_data))
    val_loss.append(epoch_loss/len(val_data))
    curr_loss = epoch_loss/len(val_data)
    if(curr_loss < best_loss):
      best_loss = curr_loss 
      best_model_state_dict = {k:v for k, v in model.state_dict().items()}
      best_model_state_dict = OrderedDict(best_model_state_dict)
    
  fig= plt.figure(figsize=(20,10))
  plt.plot(np.array(running_epochs), np.array(running_loss), 'b', label='Training loss')
  plt.plot(np.array(running_epochs), np.array(val_loss), 'g', label = 'validation loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()
  
  fig= plt.figure(figsize=(20,10))
  plt.plot(np.array(running_epochs), np.array(train_accuracy), 'b', label='Training accuracy')
  plt.plot(np.array(running_epochs), np.array(val_accuracy), 'g', label = 'validation accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.show()
  
  model.load_state_dict(best_model_state_dict)
  print("Accuracy of the Model on Train Data :",evaluate(model, train_data))
  print("Accuracy of the Model on Validation Data :",evaluate(model, val_data))
  print("Accuracy of the Model on Test Data :",evaluate(model, test_data, 1))
    