# フレームの重要度をフレームの重みで決定し記憶していく

# --- import section --- #
import os
import h5py
import math
import random
import numpy as np
import torch
import torch.nn as nn
import DebugFunc as df

from data import get_eval_loader

import sys
args = sys.argv

# --- Variable declaration section --- #
feats_path = "./AllKeyVideos/"
#video_name = "BG_335"
#video_name = "BG_22598"
#video_name = "BG_35841"
video_name = args[1]
h5_file_path = "_features.h5"
feature_h5_path = feats_path + video_name + h5_file_path
feature_h5_feats = 'feats'
feature_h5_lens = 'lens'
train_range = (0, 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float
txt_file = "text_file/" + video_name + "_key.txt"
final_txt = "text_file/" + video_name + "_final_key" + ".txt"
feat_txt = "text_file/" + video_name + "_feat_key" + ".txt"
batch_size = 1
num_epochs = 10
final_reward = 0
d_max = 0
d_min = 0
max_frame = 0

# --- Code section --- #

f = open(txt_file, "w")
text = "Video" + "," + "Frame" + "," + "Action" + "," + "Reward" +"\n"
f.writelines(text)
ff = open(final_txt, "w")
ftxt = "Num" + "," + "Reward" + "\n"
ff.writelines(ftxt)
#f3 = open(feat_txt, "w")
#f3txt = "frame" + "," + "feature:a"+","+"feature:b"+","+"cossim:c" + "\n"
#f3.writelines(f3txt)

feature = get_eval_loader(train_range, feature_h5_path)

for i, (vid_f, video_ids) in enumerate(feature):
    vid_f = vid_f.reshape((vid_f.shape[1], vid_f.shape[2]))
    vid_f = vid_f / torch.sqrt(torch.unsqueeze(torch.sum(vid_f**2, axis=1), 1))
    vid_f = torch.unsqueeze(vid_f, 0)
    vid_f = vid_f.to(device)
    max_frame = vid_f.shape[1]
    for fr in range(max_frame-1):
        a = vid_f[batch_size-1, fr]
        b = vid_f[batch_size-1, fr+1]
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        c = cos(a, b)
        #print("a:{}".format(a))
        #print("b:{}".format(b))
        #print("c:{}".format(c))
        #print(c)
        #if fr == 3:
            #for a_size in range(2048):
                #f3txt = str(fr) + "," + str(a[a_size].item())+","+str(b[a_size].item())+","+str(c.item())+"\n"
                #f3.writelines(f3txt)
        #f3txt = str(fr) + "," + str(a)+","+str(b)+","+str(c)+"\n"
        #f3.writelines(f3txt)
        if fr == 0:
            stack_a = a
            stack_b = b
            #print(stack_a.shape)
            d_max = c
            d_min = c
        if d_max < c:
            d_max = c
        if d_min > c:
            d_min = c

#f3.close()
#print("フレーム間の最大類似度:{}".format(d_max))
#print("フレーム間の最小類似度:{}".format(d_min))
# rand_action = np.random.rand(max_frame)
# rand_action[0] = 1.0
# print(rand_action)
#print("!!!!!!!!!!!!!!!!!!!!!!!!")
for i, (videos_src, video_ids) in enumerate(feature):
    """
    videos_src2 = videos_src.reshape((videos_src.shape[1], videos_src.shape[2]))
    videos = videos_src2 / torch.sqrt(torch.unsqueeze(torch.sum(videos_src2**2, axis=1), 1))
    sims = torch.mm(videos, videos.t())
    sims.fill_diagonal_(-100)
    sim_max = sims.max()
    sims.fill_diagonal_(100)
    sim_min = sims.min()
    videos = torch.unsqueeze(videos, 0)
    videos = videos.to(device)
    """
    videos_src2 = videos_src.reshape((videos_src.shape[1], videos_src.shape[2]))
    videos = videos_src2 / torch.sqrt(torch.unsqueeze(torch.sum(videos_src2**2, axis=1), 1))
    videos = torch.unsqueeze(videos, 0)
    videos = videos.to(device)
    

    V_p = torch.zeros(2048, dtype=dtype).to(device)
    stack_reward = 0
    for frame in range(max_frame-1):
        #print("---- {}".format(frame))
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        v_t = videos[batch_size-1, frame]
        v_next = videos[batch_size-1, frame+1]
        #print(v_t.shape)
        #if stack_a.sum() == v_t.sum():
            #print("Foo")
        #print("aaa{}".format(d_min))
        #print("bbb{}".format(d_max))
        action_inv = (cos(v_t, v_next) - d_min) / (d_max - d_min)
        #if frame == 0:
            #print(cos(v_t, v_next).item())
            #print(d_min.item())
            #print(d_max.item())
            #print(action_inv.item())
        #if cos(v_t, v_next).item() == d_min.item():
            #print("aAA")
            #print(action_inv)
        #if cos(v_t, v_next) == d_max:
            #print("bBB")
            #print(action_inv)
        #print(action_inv)
        action = 1 - action_inv
        if frame == 0:
            action = 1.0
        #print(action)
        sum_feature = V_p + action*v_t
        sum_feature_norm = torch.norm(sum_feature)
        V_p = sum_feature/sum_feature_norm
        # cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        reward = cos(V_p, v_next)
        # df.set_trace()
        """
        reward = (reward - sim_min) / (sim_max - sim_min)
        """
        stack_reward += reward
        text = str(i) + "," + str(frame) + "," + str(action) + "," + str(reward.item()) + "\n"
        f.writelines(text)
    stack_reward = stack_reward / (max_frame-1)
    ftxt = str(0) + "," + str(stack_reward.item()) + "\n"
    ff.writelines(ftxt)
    final_reward += stack_reward
final_reward = final_reward
ftxt = "final" + "," + str(final_reward.item()) + "\n"
ff.writelines(ftxt)

print(final_reward.item())

f.close()
ff.close()
print("Fin!!")
