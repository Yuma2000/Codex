#!/usr/bin/env python
# coding: utf-8

# ## 事前準備
# 
# 1. 必要なモジュール・ファイルのインストール
# 2. LSTMAutoEncoderのモデルの構築

# In[1]:


import torch
import numpy as np
from torch import nn, optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import rc
import glob
import os
import sys
import copy
import time
from dataLoader import get_eval_loader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[2]:


##LSTM AutoEncoder

#---Encoder---#
class Encoder(nn.Module):
    def __init__(self, n_features, embedding_dim=64):
        super(Encoder, self).__init__()

        #self.seq_len, self.n_features = seq_len, n_features
        self.n_features = n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
    
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.m1 = nn.Linear(2048, self.hidden_dim)
        self.m2 = nn.Linear(1024, 1)
        self.lstm1 = nn.LSTM(
            input_size=2048,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
    def forward(self, x):
        y = x.shape[1]
        x_rnn1, (h_rnn1, c_rnn1) = self.rnn1(x)
        #print("LSTM1 output.shape : {}".format(x.shape))
        #print("LSTM1 output : {}".format(x))
        #print("h_check : {}".format(h_check))
        #print("c_check : {}".format(c_check))
        x, (hidden_n, _) = self.rnn2(x_rnn1)
        return hidden_n, y, h_rnn1, c_rnn1, x_rnn1

#---Decoder---#
class Decoder(nn.Module):
    def __init__(self, input_dim=64, n_features=2048):
        
        super(Decoder, self).__init__()
        #self.seq_len, self.input_dim = seq_len, input_dim
        self.input_dim = input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)
        
        self.o1 = nn.Linear(self.hidden_dim, 2048)
        self.lstm1 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.n_features,
            num_layers=1,
            batch_first=True
        )
        
    def forward(self, x, y):
        x = x.repeat(1, y, 1)
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        return self.output_layer(x)

#---RecurrentAutoEncoder---#
class RecurrentAutoencoder(nn.Module):
    def __init__(self, n_features, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(n_features, embedding_dim).to(device)
        self.decoder = Decoder(embedding_dim, n_features).to(device)
        
    def forward(self, x):
        x, y, h_ra, c_ra, x_ra = self.encoder(x)
        x = self.decoder(x, y)
            
        return x, h_ra, c_ra, x_ra


# ## LSTMAutoEncoderに学習済みパラメータをIN

# In[3]:


# 隠れ状態・メモリセル取得用model
model1 = RecurrentAutoencoder(2048, 128)
model1 = model1.to(device)
MODEL_PATH1 = 'LSTM_AutoEncoder_Epoch_10000_lr_1e-5.pth'
model1 = torch.load(MODEL_PATH1)


# ## LSTMAutoEncoderから隠れ状態・メモリセル取得
# 
# 映像を一本用いて取得する．
# 
# 隠れ状態 → h_pre
# 
# メモリセル → c_pre

# In[4]:


optimizer = torch.optim.Adam(model1.parameters(), lr=1e-5)
criterion = nn.L1Loss(reduction='mean').to(device)
train_range = (0,1)
for filepath_list1 in glob.glob("./Features/AllKeyVideos/BG_*_features.h5"):
    print("Video Path : {}".format(filepath_list1))
    feature1 = get_eval_loader(train_range, filepath_list1)
    model1.train()
    #model1.eval()
    for i1, (videos1, video_ids1) in enumerate(feature1):
        optimizer.zero_grad()
        videos1 = videos1.to(device)
        seq_pred_pre, h_pre, c_pre, x_pre = model1(videos1)
        loss1 = criterion(seq_pred_pre, videos1)
        loss1.backward()
        optimizer.step()
        print(h_pre)
        print(h_pre.shape)
        print(x_pre)
        print(x_pre.shape)
    break


# ## パラメータ取得

# In[5]:


whi = model1.encoder.rnn1.weight_hh_l0[:256]
whf = model1.encoder.rnn1.weight_hh_l0[256:512]
whg = model1.encoder.rnn1.weight_hh_l0[512:768]
who = model1.encoder.rnn1.weight_hh_l0[768:1024]

wii = model1.encoder.rnn1.weight_ih_l0[:256]
wif = model1.encoder.rnn1.weight_ih_l0[256:512]
wig = model1.encoder.rnn1.weight_ih_l0[512:768]
wio = model1.encoder.rnn1.weight_ih_l0[768:1024]

bii = model1.encoder.rnn1.bias_ih_l0[:256]
bif = model1.encoder.rnn1.bias_ih_l0[256:512]
big = model1.encoder.rnn1.bias_ih_l0[512:768]
bio = model1.encoder.rnn1.bias_ih_l0[768:1024]

bhi = model1.encoder.rnn1.bias_hh_l0[:256]
bhf = model1.encoder.rnn1.bias_hh_l0[256:512]
bhg = model1.encoder.rnn1.bias_hh_l0[512:768]
bho = model1.encoder.rnn1.bias_hh_l0[768:1024]
print(whi)
print(whi.shape)
print(bhi.shape)


# ## 新たなLSTMAutoEncoderパラメータ取得

# In[6]:


model_params1 = model1.state_dict()


# ## LSTMパラメータ更新

# In[7]:


# パラメータ再更新・比較用隠れ状態・メモリセル取得用
model2 = RecurrentAutoencoder(2048, 128)
model2 = model2.to(device)

model2.load_state_dict(model_params1)
#model2 = torch.load(model_params1)


# ## 新たなパラメータで2回目

# In[8]:


train_range = (0,1)

for filepath_list2 in glob.glob("./Features/AllKeyVideos/BG_*_features.h5"):
    print("Video Path : {}".format(filepath_list2))
    feature2 = get_eval_loader(train_range, filepath_list2)
    model2.eval()
    for i2, (videos2, video_ids2) in enumerate(feature2):
        videos2 = videos2.to(device)
        #seq_len = videos.shape[1]
        #n_feature = videos.shape[2]
        seq_pred_val, h_val, c_val, x_val = model2(videos2)
        print(h_val)
    break


# In[9]:


train_range = (0,1)

for filepath_list5 in glob.glob("./Features/AllKeyVideos/BG_*_features.h5"):
    print("Video Path : {}".format(filepath_list5))
    feature5 = get_eval_loader(train_range, filepath_list5)
    model1.eval()
    for i5, (videos5, video_ids5) in enumerate(feature5):
        videos5 = videos5.to(device)
        #seq_len = videos.shape[1]
        #n_feature = videos.shape[2]
        seq_pred_val1, h_val1, c_val1, x_val1 = model1(videos5)
        print(h_val1)
    break


# ## 隠れ状態・メモリセル・一応出力の獲得（比較用）

# In[10]:


# h_val, c_val, x_val
h_val.shape


# ## 1回目のLSTMAutoEncoderのパラメータを用いて新たな隠れ状態・メモリセルの獲得

# In[12]:


train_range = (0,1)
for filepath_list3 in glob.glob("./Features/AllKeyVideos/BG_*_features.h5"):
    print(filepath_list3)
    feature3 = get_eval_loader(train_range, filepath_list3)
    break

##h_t = h_pre.squeeze(0).squeeze(0)
##c_t = c_pre.squeeze(0).squeeze(0)

h_t = torch.zeros(256).to(device)
print(h_t.shape)
c_t = torch.zeros(256).to(device)

for i3, (videos3, video_ids3) in enumerate(feature3):
    videos3 = videos3.to(device)
    for i4 in range(videos3.shape[1]):
        
        g = torch.tanh(torch.mm(wig,videos3[0][i4].unsqueeze(1)).squeeze(1) + torch.mm(whg,h_t.unsqueeze(1)).squeeze(1) + bhg + big)
        i = torch.sigmoid(torch.mm(wii,videos3[0][i4].unsqueeze(1)).squeeze(1) + torch.mm(whi,h_t.unsqueeze(1)).squeeze(1) + bhi + bii)
        f = torch.sigmoid(torch.mm(wif,videos3[0][i4].unsqueeze(1)).squeeze(1) + torch.mm(whf,h_t.unsqueeze(1)).squeeze(1) + bhf + bif)
        o = torch.sigmoid(torch.mm(wio,videos3[0][i4].unsqueeze(1)).squeeze(1) + torch.mm(who,h_t.unsqueeze(1)).squeeze(1) + bho + bio)
        
        c_t = f*c_t + i*g
        h_t = o*torch.tanh(c_t)

print(h_t)


# ## 計算で獲得した隠れ状態・メモリセル・出力の確認

# In[13]:


# h_t, c_t, x_t?
h_t


# ## 計算の結果と2回目の学習の結果の比較

# In[14]:


# 隠れ状態
h_difference = (torch.abs(h_val.squeeze(0).squeeze(0)))-(torch.abs(h_t))
print(h_difference)
h_max = h_difference.max()
h_min = h_difference.min()


# In[15]:


# メモリセル
c_difference = (torch.abs(c_val.squeeze(0).squeeze(0))-(torch.abs(c_t)))
print(c_difference)
c_max = c_difference.max()
c_min = c_difference.min()


# In[16]:


h_max.item(), h_min.item(), c_max.item(), c_min.item()


# In[ ]:




