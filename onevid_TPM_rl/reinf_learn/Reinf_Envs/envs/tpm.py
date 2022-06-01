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
        self.action_space = gym.spaces.Discrete(11)
        self.observation_space = self.V_p
        self.frame_max = 20
        self.time = 0
        self.profit = 0
        self.done = False
        self.reward_range = [-1., 1.]
        #self.reward_range = [0., 1.]
        self.cos_max = 1
        self.cos_min = -1
        self.reset(1, -1)

    def reset(self, sim_max, sim_min):
        # 諸々の変数を初期化する
        # self.V_p = torch.zeros(2048, device=device, dtype=dtype)
        self.V_p = torch.zeros(2048, dtype=dtype).to(device)
        self.time = 0
        self.profit = 0
        self.done = False
        self.cos_max = sim_max
        self.cos_min = sim_min
        return self.observe(self.V_p)

    def step(self, action, V_t, v_next):
        # 1ステップ進める処理を記述。戻り値は observation, reward, done(ゲーム終了したか), info(追加の情報の辞書)
        
        V_p = self.pool(action, self.V_p, V_t)
        reward = self.get_reward(V_p, v_next)
        self.profit += reward
        #print("step : {}/{} reward : {}".format(self.time,self.frame_max, reward))
        if self.time == self.frame_max-1:
            #print("profit : {}".format(self.profit))
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

    def get_reward(self, V_p, V_t):
        #ここで次のフレームと記憶をcos類似度で報酬を得る
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        #V_p_tensor = torch.from_numpy(V_p)
        #V_t_tensor = torch.from_numpy(V_t)
        #print("V_t : {}".format(V_t))
        reward = cos(V_p, V_t)
        #print("reward : {}".format(reward))
        ###-----###
        reward = (reward - self.cos_min) / (self.cos_max - self.cos_min)
        return reward

    def observe(self, V_p):
        # マップに勇者の位置を重ねて返す
        observation = V_p
        self.V_p = V_p
        return observation

    def pool(self, action, V_p, V_t): # 特徴をpoolingする
        
        #ここに比率によるif文を入れる
        ratio = 0.0
        if action == 0:
            ratio = 0.0
        elif action == 1:
            ratio = 0.1
        elif action == 2:
            ratio = 0.2
        elif action == 3:
            ratio = 0.3
        elif action == 4:
            ratio = 0.4
        elif action == 5:
            ratio = 0.5
        elif action == 6:
            ratio = 0.6
        elif action == 7:
            ratio = 0.7
        elif action == 8:
            ratio = 0.8
        elif action == 9:
            ratio = 0.9
        elif action == 10:
            ratio = 1.0
        
        #V_p_tensor = torch.from_numpy(V_p)
        #print("action : {}".format(action))

        
        #sum_feature = V_p + ratio*V_t
        #V_p = sum_feature / (1+ratio)

        # if self.time == 0:
        #    ratio = 1.0
            
        sum_feature = V_p + ratio*V_t
        sum_feature_norm = torch.norm(sum_feature)
        #if sum_feature_norm == 0.5:
        #    print("sum_feature : {}".format(sum_feature))
        #    print("V_p : {}".format(V_p))
        #    print("ratio : {}".format(ratio))
        #    print("V_t : {}".format(V_t))
        #    dp.set_trace()
        #print("sum_f : {}".format(sum_feature_norm))
        ##====##
        #if action == 0:
        #    V_p = sum_feature
        #else:
        #    V_p = sum_feature/sum_feature_norm

        V_p = sum_feature/sum_feature_norm
        
        return V_p