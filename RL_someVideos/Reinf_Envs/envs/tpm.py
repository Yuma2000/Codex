# -----import section----- #
import sys
import gym
import numpy as np
import gym.spaces
import torch
import torch.nn as nn
import DebugFunc as dp
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float

class TimePM(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        super().__init__()
        # action_space, observation_space, reward_range を設定する
        self.V_p = torch.zeros(2048, dtype=dtype).to(device)
        self.action_space = gym.spaces.Discrete(21)
        self.observation_space = self.V_p
        self.frame_max = 20
        self.time = 0
        self.profit = 0
        self.done = False
        self.reward_range = [-1., 1.]
        self.cos_max = 1
        self.cos_min = -1
        self.reset(1, -1)

    def reset(self, sim_max, sim_min):
        # 諸々の変数を初期化する
        self.V_p = torch.zeros(2048, dtype=dtype).to(device)
        self.time = 0
        self.profit = 0
        self.done = False
        self.cos_max = sim_max
        self.cos_min = sim_min
        return self.observe(self.V_p)
    
    """
    # 強化学習のステップ
    ## action => model.pyの出力層のargmaxの値 
    ## V_t => 記憶に追加するフレーム
    ## V_next => 追加したフレームの次のフレーム
    ##  => 
    ## 
    ## 
    """
    def step(self, action, V_t, v_next):
        #
        V_p = self.pool(action, self.V_p, V_t)
        reward = self.get_reward(V_p, v_next)
        self.profit += reward
        if self.time == self.frame_max-1:
            self.done = True
        observation = self.observe(V_p)
        self.time += 1

        return observation, reward, self.done, {}


    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    """
    # cos類似度による報酬の計算
    ## V_p => 記憶
    ## V_t => 追加したフレームの次のフレーム
    ## 
    """
    def get_reward(self, V_p, V_next):
        #
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        reward = cos(V_p, V_next)
        ##
        # reward = (reward - self.cos_min) / (self.cos_max - self.cos_min)
        ##
        return reward

    """
    # 変数の更新
    """
    def observe(self, V_p):
        #
        observation = V_p
        self.V_p = V_p
        #
        return observation

    """
    # 記憶への蓄積を行う
    ## action => model.pyの出力層のargmaxの値(0~20の21種類)
    ## V_p => 記憶
    ## V_t => t番目のフレームの特徴ベクトル
    ## 
    """
    def pool(self, action, V_p, V_t):
        #
        ratio = 0.0
        ratio = action * 0.05 #21
        sum_feature = V_p + ratio*V_t
        sum_feature_norm = torch.norm(sum_feature)
        V_p = sum_feature/sum_feature_norm
        #
        return V_p
