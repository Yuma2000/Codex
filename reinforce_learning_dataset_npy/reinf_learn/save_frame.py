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
import cv2
import os
import DebugFunc as df
import sys

feats_path = "../dataset_toolkit/feats_key/"
video_name = "BG_335"
#video_name = "BG_22598"
#video_name = "BG_35841"
h5_file_path = "_features.h5"
feature_h5_path = feats_path + video_name + h5_file_path
key_frame = "./frame_extraction/"+video_name+"_keyframeIDs.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_frame = True#False # True

with open(key_frame) as f0:
    lines = f0.readlines()

if save_frame == True:
    print("フレームの保存開始")
    v_path = "/home/kouki/remote-mount/tv2009/devel08/video/" + video_name + ".mpg"
    cap = cv2.VideoCapture(v_path)
    for s_frame in range(0,int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1):
        for b in range(len(lines)):
            if s_frame == int(lines[b]):
                cap.set(cv2.CAP_PROP_POS_FRAMES, s_frame)
                ret, frame = cap.read()
                save_path = "./save_frame_key/" + video_name + "/" + str(s_frame) + ".png"
                os.makedirs(os.path.dirname(save_path), exist_ok = True)
                if ret:
                    cv2.imwrite(save_path,frame)
    print("フレームの保存終了")
