# LSTMを使用し、取得したinput gate(i)とforget gate(f)の値を使用
# actionを決定するwを||i||l2/||f||l2で求め、記憶V + w*frame_feat --(norm)-> new 記憶V

# --- import section --- #
import os
import h5py
import math
import random
import numpy as np
import torch
import DebugFunc as df
from data import get_eval_loader
#from model_lstm import RecurrentAutoencoder
import sys
args = sys.argv

from torch import nn, optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import rc
import glob
import copy
import time

# --- Variable declaration section --- #
feats_path = "./AllKeyVideos/"
video_name = args[1]
print("Video Name : {}".format(video_name))
feature_h5_path = feats_path + video_name + "_features.h5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_range = (0,1)
batch_size = 1
num_epochs = 10
final_reward = 0
d_max = 0
d_min = 0
max_frame = 0
dtype = torch.float
txt_file1 = "text_file/1LSTM/" + video_name +"_1LSTM"+ "_key.txt"
txt_file2 = "text_file/1LSTM/" + video_name +"_1LSTM"+ "_final_key" + ".txt"
txt_file3 = "text_file/1LSTM/" + video_name +"_1LSTM"+ "_best_action" + ".txt"

f1 = open(txt_file1, "w")
text1 = "Video" + "," + "Frame" + "," + "Action" + "," + "Reward" +"\n"
f1.writelines(text1)
f2 = open(txt_file2, "w")
text2 = "Num" + "," + "Reward" + "\n"
f2.writelines(text2)
f3 = open(txt_file3, "w")
text3 = "Video" + "," + "Frame" + "," + "LSTMAction" + "\n"
f3.writelines(text3)

# --- LSTM --- #

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
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        y = x.shape[1]
        x_rnn1, (h_rnn1, c_rnn1) = self.rnn1(x)
        #x, (hidden_n, _) = self.rnn2(x_rnn1)
        return h_rnn1, y


#---Decoder---#
class Decoder(nn.Module):
    def __init__(self, input_dim=64, n_features=2048):

        super(Decoder, self).__init__()
        #self.seq_len, self.input_dim = seq_len, input_dim
        self.input_dim = input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features

        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x, y):
        x = x.repeat(1, y, 1)
        x, (hidden_n, cell_n) = self.rnn1(x)
        #x, (hidden_n, cell_n) = self.rnn2(x)
        return self.output_layer(x)

#---RecurrentAutoEncoder---#
class RecurrentAutoencoder(nn.Module):
    def __init__(self, n_features, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(n_features, embedding_dim).to(device)
        self.decoder = Decoder(embedding_dim, n_features).to(device)

    def forward(self, x):
        x, y = self.encoder(x)
        x = self.decoder(x, y)

        return x



# --- LSTM c & t get section --- #
model1 = RecurrentAutoencoder(2048, 128)
model1 = model1.to(device)
MODEL_PATH1 = './LSTM_model/LSTM_AutoEncoder_1LSTM_Epoch_10000_lr_1e-5.pth'
model1 = torch.load(MODEL_PATH1)

optimizer = torch.optim.Adam(model1.parameters(), lr=1e-5)
criterion = nn.L1Loss(reduction='mean').to(device)

feature1 = get_eval_loader(train_range, feature_h5_path)
model1.train()

for i1, (videos1_src, video_ids1) in enumerate(feature1):
    videos1_src2 = videos1_src.reshape((videos1_src.shape[1], videos1_src.shape[2]))
    videos1 = videos1_src2 / torch.sqrt(torch.unsqueeze(torch.sum(videos1_src2**2, axis=1), 1))
    videos1 = torch.unsqueeze(videos1, 0)
    videos1 = videos1.to(device)
    optimizer.zero_grad()
    seq_pred_pre = model1(videos1)
    loss1 = criterion(seq_pred_pre, videos1)
    loss1.backward()
    optimizer.step()


# --- LSTM weight & bias --- #
whi = model1.encoder.rnn1.weight_hh_l0[:128]
whf = model1.encoder.rnn1.weight_hh_l0[128:256]
whg = model1.encoder.rnn1.weight_hh_l0[256:384]
who = model1.encoder.rnn1.weight_hh_l0[384:512]

wii = model1.encoder.rnn1.weight_ih_l0[:128]
wif = model1.encoder.rnn1.weight_ih_l0[128:256]
wig = model1.encoder.rnn1.weight_ih_l0[256:384]
wio = model1.encoder.rnn1.weight_ih_l0[384:512]

bii = model1.encoder.rnn1.bias_ih_l0[:128]
bif = model1.encoder.rnn1.bias_ih_l0[128:256]
big = model1.encoder.rnn1.bias_ih_l0[256:384]
bio = model1.encoder.rnn1.bias_ih_l0[384:512]

bhi = model1.encoder.rnn1.bias_hh_l0[:128]
bhf = model1.encoder.rnn1.bias_hh_l0[128:256]
bhg = model1.encoder.rnn1.bias_hh_l0[256:384]
bho = model1.encoder.rnn1.bias_hh_l0[384:512]
# --- LSTM f & i get section--- #
model_params1 = model1.state_dict()
model2 = RecurrentAutoencoder(2048, 128)
model2 = model2.to(device)
model2.load_state_dict(model_params1)
feature3 = get_eval_loader(train_range, feature_h5_path)

h_t = torch.zeros(128).to(device)
print(h_t.shape)
c_t = torch.zeros(128).to(device)

w_stack = []

for i3, (videos3_src, video_ids3) in enumerate(feature3):
    videos3_src2 = videos3_src.reshape((videos3_src.shape[1], videos3_src.shape[2]))
    videos3 = videos3_src2 / torch.sqrt(torch.unsqueeze(torch.sum(videos3_src2**2, axis=1), 1))
    videos3 = torch.unsqueeze(videos3, 0)
    videos3 = videos3.to(device)
    for i4 in range(videos3.shape[1]):

        g_gate = torch.tanh(torch.mm(wig,videos3[0][i4].unsqueeze(1)).squeeze(1) + torch.mm(whg,h_t.unsqueeze(1)).squeeze(1) + bhg + big)
        i_gate = torch.sigmoid(torch.mm(wii,videos3[0][i4].unsqueeze(1)).squeeze(1) + torch.mm(whi,h_t.unsqueeze(1)).squeeze(1) + bhi + bii)
        f_gate = torch.sigmoid(torch.mm(wif,videos3[0][i4].unsqueeze(1)).squeeze(1) + torch.mm(whf,h_t.unsqueeze(1)).squeeze(1) + bhf + bif)
        o_gate = torch.sigmoid(torch.mm(wio,videos3[0][i4].unsqueeze(1)).squeeze(1) + torch.mm(who,h_t.unsqueeze(1)).squeeze(1) + bho + bio)

        i_stack = torch.sqrt(torch.sum(i_gate*i_gate))
        f_stack = torch.sqrt(torch.sum(f_gate*f_gate))
        w_stack.append(i_stack/f_stack)

        c_t = f_gate*c_t + i_gate*g_gate
        h_t = o_gate*torch.tanh(c_t)

print("LSTM Action len : {}".format(len(w_stack)))
#print("LSTM Action : {}".format(w_stack))


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
    action = 0
    max_frame = videos.shape[1]
    print("Max frame = {}".format(max_frame))

    for frame in range(max_frame-1):
        #print("---- {}".format(frame))
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        reward = 0.0
        #action = 0.0
        v_t = videos[batch_size-1, frame]
        v_next = videos[batch_size-1, frame+1]

        action = w_stack[frame]
        sum_feature = V_p + action*v_t
        sum_feature_norm = torch.norm(sum_feature)
        V_p = sum_feature/sum_feature_norm
        reward = cos(V_p, v_next)
        """
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
        """
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







