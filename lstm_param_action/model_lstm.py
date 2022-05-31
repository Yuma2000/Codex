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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


