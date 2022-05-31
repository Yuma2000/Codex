# フレームの重要度をRandomで決定し記憶していく

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

# --- Variable declaration section --- #
feats_path = "./feats/"
# video_name = "BG_335"
video_name = "BG_22598"
# video_name = "BG_35841"
h5_file_path = "_features.h5"
feature_h5_path = feats_path + video_name + h5_file_path
feature_h5_feats = 'feats'
feature_h5_lens = 'lens'
train_range = (0, 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float
txt_file = "text_file/" + video_name + ".txt"
final_txt = "text_file/" + video_name + "_final" + ".txt"
batch_size = 1
num_epochs = 10
final_reward = 0

# --- Code section --- #

f = open(txt_file, "w")
text = "Video" + "," + "Frame" + "," + "Action" + "," + "Reward" +"\n"
f.writelines(text)
ff = open(final_txt, "w")
ftxt = "Num" + "," + "Reward" + "\n"
ff.writelines(ftxt)


feature = get_eval_loader(train_range, feature_h5_path)

for i, (videos, video_ids) in enumerate(feature):
    max_frame = videos.shape[1]

# rand_action = np.random.rand(max_frame)
# rand_action[0] = 1.0
# print(rand_action)

for num in range(num_epochs):
    print(num)
    rand_action = np.random.rand(max_frame)
    rand_action[0] = 1.0
    for i, (videos_src, video_ids) in enumerate(feature):
        videos_src2 = videos_src.reshape((videos_src.shape[1], videos_src.shape[2]))
        videos = videos_src2 / torch.sqrt(torch.unsqueeze(torch.sum(videos_src2**2, axis=1), 1))
        sims = torch.mm(videos, videos.t())
        sims.fill_diagonal_(-100)
        sim_max = sims.max()
        sims.fill_diagonal_(100)
        sim_min = sims.min()
        videos = torch.unsqueeze(videos, 0)
        videos = videos.to(device)
        V_p = torch.zeros(2048, dtype=dtype).to(device)
        stack_reward = 0
        for frame in range(max_frame-1):
            print("---- {}".format(frame))
            v_t = videos[batch_size-1, frame]
            v_next = videos[batch_size-1, frame+1]
            sum_feature = V_p + rand_action[frame]*v_t
            sum_feature_norm = torch.norm(sum_feature)
            V_p = sum_feature/sum_feature_norm
            cos = nn.CosineSimilarity(dim=0, eps=1e-6)
            reward = cos(V_p, v_next)
            reward = (reward - sim_min) / (sim_max - sim_min)
            stack_reward += reward
            text = str(i) + "," + str(frame) + "," + str(rand_action[frame]) + "," + str(reward.item()) + "\n"
            f.writelines(text)
        stack_reward = stack_reward / (max_frame-1)
        ftxt = str(num) + "," + str(stack_reward) + "\n"
        ff.writelines(ftxt)
        final_reward += stack_reward
final_reward = final_reward / num_epochs
ftxt = "final" + "," + str(final_reward) + "\n"
ff.writelines(ftxt)

print(final_reward)

f.close()
ff.close()
print("Fin!!")
