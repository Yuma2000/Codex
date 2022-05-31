import gym
import math
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as T
import Reinf_Envs
import pickle
import h5py
from conf import train_caption_pkl_path
from conf import feature_h5_path, feature_h5_feats
from conf import train_range
from conf import device , dtype
import os
import DebugFunc as df

class DQN(nn.Module):

    def __init__(self, V_p, V_t, outputs=1):
        super(DQN, self).__init__()
        self.input = nn.Linear(V_p+V_t, outputs)  # 1024)
        """
        self.input = nn.Linear(V_p+V_t, 1024)
        
        self.FC1024_1 = nn.Linear(1024, 1024)
        self.FC1024_2 = nn.Linear(1024, 1024)

        self.FC_1024_512 = nn.Linear(1024, 512)

        self.FC_512_64 = nn.Linear(512, 64)

        self.output = nn.Linear(64, 11)
        #self.softplus = nn.Softplus()
        #self.dropout = nn.Dropout(p = 0.45)

        self.inp = nn.Linear(4096, 1024)
        self.act = nn.Linear(1024, 11)
        """
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        """
        x = self.input(x)
        x = self.softplus(x)

        x = self.FC1024_1(x)
        x = self.softplus(x)
        x = self.dropout(x)

        x = self.FC1024_2(x)
        x = self.softplus(x)
        x = self.dropout(x)

        x = self.FC_1024_512(x)
        x = self.softplus(x)
        x = self.dropout(x)

        x = self.FC_512_64(x)
        x = self.softplus(x)

        action = self.output(x)
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

        #x = self.inp(x)
        #x = self.softplus(x)
        #action = self.act(x)
        action = self.input(x)
        return action