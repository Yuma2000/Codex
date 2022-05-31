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
print("Video Name : {}".format(video_name))
h5_file_path = "_features.h5"
feature_h5_path = feats_path + video_name + h5_file_path
feature_h5_feats = 'feats'
feature_h5_lens = 'lens'
train_range = (0, 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float
txt_file1 = "text_file/" + video_name + "_key.txt"
txt_file2 = "text_file/" + video_name + "_final_key" + ".txt"
txt_file3 = "text_file/" + video_name + "_best_action" + ".txt"
batch_size = 1
num_epochs = 10
final_reward = 0
d_max = 0
d_min = 0
max_frame = 0

# --- Code section --- #

f1 = open(txt_file1, "w")
text1 = "Video" + "," + "Frame" + "," + "Action" + "," + "Reward" +"\n"
f1.writelines(text1)
f2 = open(txt_file2, "w")
text2 = "Num" + "," + "Reward" + "\n"
f2.writelines(text2)
f3 = open(txt_file3, "w")
text3 = "Video" + "," + "Frame" + "," + "BestAction" + "\n"
f3.writelines(text3)

feature = get_eval_loader(train_range, feature_h5_path)

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
    max_frame = videos.shape[1]
    print("Max frame = {}".format(max_frame))
    for frame in range(max_frame-1):
        #print("---- {}".format(frame))
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        reward = 0.0
        #action = 0.0
        v_t = videos[batch_size-1, frame]
        v_next = videos[batch_size-1, frame+1]
        if frame == 0:
            action = 1.0
            sum_feature = V_p + action*v_t
            sum_feature_norm = torch.norm(sum_feature)
            V_p = sum_feature/sum_feature_norm
            reward = cos(V_p, v_next)
        else:
            for acts in range(21):
                action_sub = acts*0.05
                #print("Action_sub : {}".format(action_sub))
                sum_feature = V_p + action_sub*v_t
                sum_feature_norm = torch.norm(sum_feature)
                V_p_sub = sum_feature/sum_feature_norm
                reward_sub = cos(V_p_sub, v_next)
                #print(reward_sub.item())
                #print(reward)
                if reward < reward_sub:
                    reward = reward_sub
                    #print(reward.item())
                    action = action_sub
                    V_p = V_p_sub
        #print("Action : {}".format(action))
        #sum_feature = V_p + action*v_t
        #sum_feature_norm = torch.norm(sum_feature)
        #V_p = sum_feature/sum_feature_norm
        #reward = cos(V_p, v_next)
        # df.set_trace()
        #if frame == 5:
            #break
        """
        reward = (reward - sim_min) / (sim_max - sim_min)
        """
        stack_reward += reward
        text1 = str(i) + "," + str(frame) + "," + str(action) + "," + str(reward.item()) + "\n"
        f1.writelines(text1)
        text3 = str(i) + "," + str(frame) + "," + str(action) + "\n"
        f3.writelines(text3)
    stack_reward = stack_reward / (max_frame-1)
    text2 = str(0) + "," + str(stack_reward.item()) + "\n"
    f2.writelines(text2)
    final_reward += stack_reward
final_reward = final_reward
text2 = "final" + "," + str(final_reward.item()) + "\n"
f2.writelines(text2)

print(final_reward.item())

f1.close()
f2.close()
f3.close()
print("Fin!!")
