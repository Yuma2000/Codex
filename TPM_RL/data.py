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

class VideoDataset(data.Dataset):
    
    def __init__(self, eval_range, feature_h5):
        self.eval_list = tuple(range(*eval_range))
        h5_file = h5py.File(feature_h5, 'r')
        self.video_feats = h5_file[feature_h5_feats]
        #print("###")
        #print(self.video_feats[0:6,:,:])

    def __getitem__(self, index):
        video_id = self.eval_list[index]
        # print("video_id={}".format(video_id))
        video_feat = torch.from_numpy(self.video_feats[video_id])
        return video_feat, video_id

    def __len__(self):
        return len(self.eval_list)

def eval_collate_fn(data):
    '''
    用来把多个数据样本合并成一个minibatch的函数
    '''
    data.sort(key=lambda x: x[-1], reverse=True)
    videos, video_ids = zip(*data)
    # 把视频合并在一起（把2D Tensor的序列变成3D Tensor）
    videos = torch.stack(videos, 0)

    return videos, video_ids

#shuffle=True
def get_eval_loader(cap_pkl, feature_h5, batch_size=1, shuffle=False, num_workers=0, pin_memory=True):
    vd = VideoDataset(cap_pkl, feature_h5)
    # data_loader = torch.utils.data.DataLoader(dataset=vd,
    #                                          batch_size=batch_size,
    #                                          shuffle=shuffle,
    #                                          num_workers=num_workers,
    #                                          collate_fn=eval_collate_fn,
    #                                          pin_memory=pin_memory)
    data_loader = torch.utils.data.DataLoader(dataset=vd,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)
    return data_loader