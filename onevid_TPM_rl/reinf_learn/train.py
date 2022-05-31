# --- Import section --- #
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
import cv2
import os
import DebugFunc as df
import sys
import datetime
from torch.utils.tensorboard import SummaryWriter

from replayMemory import Transition
from replayMemory import ReplayMemory
from data import get_eval_loader
from model_dqn import DQN
# --- Variable declaration section --- #
feature_h5_path = "../dataset_toolkit/feats/tv_features.h5"
feature_h5_feats = 'feats'
train_range = (0, 219)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 1000
BATCH_SIZE = 128
GAMMA = 0.999
memory = ReplayMemory(100000)
max_frame = 100
batch_size = 1
views = 10000
save_frame = False
learning_rate = 1e-3#1e-3
model_save = "./model_save"
model_t_path = os.path.join(model_save, 'target_DQN.pth')
model_p_path = os.path.join(model_save, 'policy_DQN.pth')

now = datetime.datetime.now()
dir_name = "log_tensorboard/" + now.strftime('%Y%m%d_%H%M%S')
txt_file = "log_text/" + now.strftime('%Y%m%d_%H%M%S') + ".txt"

# --- Code section --- #

f = open(txt_file, "w")
text = "Epoch" + "," + "Video" + "," + "Frame" + "," + "Action" + "," + "Reward" +"\n"
f.writelines(text)

writer = SummaryWriter(log_dir = dir_name)

# 環境の読み込みと初期化
env = gym.make('TimePM-v0')
env.reset(1, -1)
# 行動数の読み込み
n_actions = env.action_space.n

# 特徴量のload
feature = get_eval_loader(train_range, feature_h5_path)

# 特徴量のサイズを入手
for i, (videos, video_ids) in enumerate(feature):
    # if i == 0:
        # print("videos.shape : {}".format(videos.shape))
    V_p_size = videos.shape[2]
    V_t_size = videos.shape[2]

# DQNのモデルの読み込みと初期化と推論モードへ移行
policy_net = DQN(V_p_size, V_t_size, n_actions).to(device)
target_net = DQN(V_p_size, V_t_size, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# オプティマイザの指定
optimizer = optim.Adam(policy_net.parameters(), lr = learning_rate)
# ステップのカウントを行う変数の宣言
steps_done = 0
# その他の変数の宣言
log = 0
log_reward = torch.empty([1]).to(device)
log_loss = torch.empty([1]).to(device)
sum_reward = 0
sum_max = 0

#-----%Actionの選択%-----#
def select_action(state, frame):
    global steps_done
    if frame == 0:
        return torch.tensor([n_actions-1], dtype=torch.long).to(device)
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).argmax().view(1)
    else:
        return torch.tensor([random.randrange(n_actions)], dtype=torch.long).to(device)
#-----%%-----#

#-----%ベルマン方程式によるQ値の計算とLossの計算、及び逆伝搬%-----#
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool
        ).to(device)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # 今のpolicy_netに通して、あのとき選択したActionの価値はなんぼかを計算している
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute the expected Q values
    # Bellmann方程式の片方で、expected_state_action_valuesがstate_action_valuesと
    # 同じになってほしい
    next_state_values = torch.zeros(BATCH_SIZE).to(device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    writer.add_scalar("loss", loss, log)

    optimizer.zero_grad()
    loss.backward()
    # 勾配のクリッピング：あまりに大きい勾配を補正する
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
#-----%%-----#

v_num_check = 0

# 学習
for view in range(views):
    print("View : {}/{}".format(view, views))
    sum_reward = 0
    for v_num, (videos_src, video_ids) in enumerate(feature):
        if v_num < v_num_check:
            continue
        # フレームの保存
        if save_frame == True:
            v_path = "./datasets/MSVD/youtube_videos/vid" + str(v_num+1) + ".avi"
            frame_count = 0
            cap = cv2.VideoCapture(v_path)
            cap2 = cv2.VideoCapture(v_path)
            while True:
                rt2, fr2 = cap2.read()
                if rt2 is False:
                    break
                frame_count += 1
            indices = np.linspace(8, frame_count - 7, max_frame, endpoint=False, dtype=int)
        # videos_srcを正規化し、その後フレーム間でのコサイン類似度のmax, minを求める
        videos_src2 = videos_src.reshape((videos_src.shape[1], videos_src.shape[2]))
        videos = videos_src2 / torch.sqrt(torch.unsqueeze(torch.sum(videos_src2**2, axis=1), 1))
        sims = torch.mm(videos, videos.t())
        sims.fill_diagonal_(-100)
        sim_max = sims.max()
        sims.fill_diagonal_(100)
        sim_min = sims.min()
        videos = torch.unsqueeze(videos, 0)
        videos = videos.to(device)

        print("----Episode : {}/{}".format(v_num,len(feature)))
        V_p = env.reset(sim_max, sim_min)
        video_sum_reward = 0
        for frame in range(max_frame-1):
            v_t = videos[batch_size-1, frame]
            state = torch.cat((V_p, v_t), dim = 0).to(device)
            action = select_action(state, frame)
            # フレームの保存
            if save_frame == True:
                frame_num = indices[frame]
                result_path = "./result_frame/" + now.strftime('%Y%m%d_%H%M%S') + \
                    "/episord_" + str(view) + "/vid_" + str(v_num+1) + \
                    "/frame_" + str(frame) + "_action_" + str(action.item()) + ".jpg"
                os.makedirs(os.path.dirname(result_path), exist_ok = True)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                rt, fr = cap.read()
                if rt:
                    cv2.imwrite(result_path, fr)
            v_next = videos[batch_size-1, frame+1]
            V_p, reward, done, _ = env.step(action, v_t, v_next)
            sum_reward += reward
            video_sum_reward += reward
            reward = torch.tensor([reward]).to(device)
            writer.add_scalar("reward", reward, log)
            next_state = torch.cat((V_p, v_next), dim = 0)
            v_dim = torch.zeros(1,4096, dtype=dtype).to(device)
            v_dim_next = torch.zeros(1,4096, dtype=dtype).to(device)
            act_dim = torch.zeros(1,1, dtype=torch.long).to(device)
            v_dim[0][:] = state
            act_dim[0][:] = action
            v_dim_next[0][:] = next_state
            memory.push(v_dim, act_dim, v_dim_next, reward)
            optimize_model()
            log += 1
            text = str(view) + "," + str(v_num) + "," + str(frame) + "," + str(action.item()) + "," + str(reward.item()) + "\n"
            f.writelines(text)
        
        # 各映像ごとの報酬
        video_sum_reward = video_sum_reward / (max_frame - 1)
        writer.add_scalar("各映像ごとの報酬の遷移", video_sum_reward, log)
        #if v_num % TARGET_UPDATE == 0:
        #    target_net.load_state_dict(policy_net.state_dict())
        
        if v_num == v_num_check:
            break
    # エピソードごとの報酬(一本の映像に対して動かした場合ここは見ないものとする)
    sum_reward = sum_reward / (len(feature)*(max_frame-1))
    writer.add_scalar("エピソードごとの報酬の遷移", sum_reward, log)
    if sum_max < sum_reward:
        sum_max = sum_reward
        torch.save(policy_net.state_dict(), model_p_path)
        torch.save(target_net.state_dict(), model_t_path)
    if view % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
       
writer.close()
f.close()
print("Fin!!")
