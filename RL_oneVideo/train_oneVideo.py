import numpy as np
import random
import math
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import gym
import sys
import os
import csv
import datetime
import glob
from collections import namedtuple
from memory_profiler import profile
import matplotlib
import matplotlib.pyplot as plt
import DebugFunc as df

import Reinf_Envs
from model import DQN, DQN_svr
from replayMemory import Transition
from replayMemory import ReplayMemory
from dataLoader import get_eval_loader

# どのGPUを使うか
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
torch.cuda.manual_seed(0)
now = datetime.datetime.now() # 現在時刻を設定
env = gym.make("TimePM-v0")
env.reset(1, -1)
n_actions = env.action_space.n
#policy_net = DQN(2048, 2048, n_actions).to(device)
#target_net = DQN(2048, 2048, n_actions).to(device)
policy_net = DQN_svr(2048, 2048, n_actions).to(device)
target_net = DQN_svr(2048, 2048, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr = 1e-4, weight_decay=0)
steps_done = 0
memory_size = 10000
memory = ReplayMemory(memory_size)
first_action = torch.tensor([n_actions-1], dtype=torch.long).to(device)

# seedは0にして使うことがほとんど
def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# 検証用の関数
def compute_qvalue(val_video_name, dfs_actions, csv_file3, epoch, csv_file4):
    action = []
    q_value = []
    q_stack = 0.0
    action_stack = []
    reward_pool = []
    ratio_list = torch.from_numpy(np.array([[i*0.1] for i in range(11)])).float().to(device)
    ratio_list[0] = 1e-7  #0にすると除算できない等の不都合があるため，小さな数に設定している．
    for feats_path in glob.glob("./Features/AllKeyVideos/" + val_video_name + "_features.h5"):
        train_range = (0,1) #動画ごとにforを回しているが，１本の動画しか用いていない．
        feature = get_eval_loader(train_range, feats_path)
        for v_num, (videos_src, video_ids) in enumerate(feature):
            videos, sim_max, sim_min = feature_norm(videos_src)
            max_frame = videos.shape[1]
            V_p = torch.zeros(2048, dtype=torch.float).to(device)
            for frame in range(max_frame-1): #フレーム数でforを回す．
                v_t = videos[0, frame]  #現在のフレームの特徴
                v_next = videos[0, frame+1]  #次のフレームの特徴
                state = torch.unsqueeze(torch.cat((V_p, v_t), dim = 0), 0)
                if frame == 0:  #初めのフレームの時
                    
                    with torch.no_grad():
                        #q_stack = target_net(state)[0][n_actions-1].item()
                        q_stack = policy_net(state)[0][n_actions-1].item()
                        action = n_actions-1
                else:
                    with torch.no_grad():
                        #q_stack = target_net(state).max(1)[0].item()
                        #action = target_net(state).max(1)[1].item()
                        q_stack = policy_net(state).max(1)[0].item()
                        action = policy_net(state).max(1)[1].item()
                        #df.set_trace()
                    
                #q_value = target_net(state).max(1)[0].detach()
                #action = target_net(state).max(1)[1].detach()
                #print(q_stack)
                mem = V_p + (torch.repeat_interleave(v_t, 11).reshape(2048,11).T)*ratio_list
                mem = mem / (torch.norm(mem, dim=1).unsqueeze(1))
                V_p = mem[action].squeeze(0)
                mem_2 = torch.mv(mem, v_next.T)
                r_norm = ((mem_2[action] - sim_min) / (sim_max - sim_min)).item()  #正規化している．

                q_value.append(q_stack)
                action_stack.append(action)
                reward_pool.append(r_norm)

                with open(csv_file3, "a") as f3:
                    data = [str(epoch), val_video_name, str(frame), str(action), str(q_stack), str(r_norm)]
                    writer = csv.writer(f3)
                    writer.writerow(data)
                
                if frame == 7:  # 深さ優先探索との比較で使用する(break)
                    break
    #print(q_value)
    e_action = np.sum(np.abs([x - y for (x, y) in zip(dfs_actions, action_stack)]))
    qvalue_average = np.mean(q_value)  #平均
    val_reward_average = np.mean(reward_pool)

    with open(csv_file3, "a") as f3:
        data = [str(epoch), val_video_name, str(-1), str(e_action), str(qvalue_average), str(val_reward_average)]
        writer = csv.writer(f3)
        writer.writerow(data)

    with open(csv_file4, "a") as f4:
        data = [str(epoch), val_video_name, str(e_action), str(qvalue_average), str(val_reward_average)]
        writer = csv.writer(f4)
        writer.writerow(data)

    #print(qvalue_average)
    return qvalue_average, e_action, val_reward_average

#モデルを保存する．
def model_save(train_video_name, key, val_video_name):
    m_save_path = "./oneVideo_result/" + "train_" + train_video_name + "-val_" + val_video_name + "/Model_save/"
    if not os.path.isdir(m_save_path):
        os.makedirs(m_save_path)
    target_path = "target_" + key + ".pth"
    policy_path = "policy_" + key + ".pth"
    model_t_path = os.path.join(m_save_path, target_path)
    model_p_path = os.path.join(m_save_path, policy_path)
    torch.save(policy_net.state_dict(), model_p_path)
    torch.save(target_net.state_dict(), model_t_path)


def fig_creat(epochs,n_loss,f_loss,f_reward,f_reward_comparison, train_video_name,action_abs,qvalue_stack,val_reward_stack,val_video_name):
    
    f_save_path = "./oneVideo_result/" + "train_" + train_video_name + "-val_" + val_video_name + "/Fig_save/"
    if not os.path.isdir(f_save_path):
        os.makedirs(f_save_path)
    reward_save_path = os.path.join(f_save_path, "reward.png")
    val_reward_save_path = os.path.join(f_save_path, "val_reward.png")
    reward_comparison_save_path = os.path.join(f_save_path, "reward_comarison.png")
    loss_save_path = os.path.join(f_save_path, "loss.png")
    arq_save_path = os.path.join(f_save_path, "action_reward_q.png")
    action_abs_save_path = os.path.join(f_save_path, "action_abs.png")
    qvalue_save_path = os.path.join(f_save_path, "qvalue.png")

    fig_title = "train_" + train_video_name + "-val_" + val_video_name

    fig1 = plt.figure(figsize=(12, 8))
    x_ax1 = np.linspace(0, epochs, epochs)
    ax1 = fig1.add_subplot(111)
    plt.title(fig_title)
    plt.plot(x_ax1, f_reward, label="reward")
    fig1.savefig(reward_save_path)

    fig2 = plt.figure(figsize=(12, 8))
    ax2 = fig2.add_subplot(111)
    plt.title(fig_title)
    plt.plot(x_ax1, f_reward_comparison, label="reward_comparison")
    fig2.savefig(reward_comparison_save_path)

    fig3 = plt.figure(figsize=(12, 8))
    x_ax3 = np.linspace(0, n_loss, n_loss)
    ax3 = fig3.add_subplot(111)
    plt.title(fig_title)
    plt.plot(x_ax3, f_loss, label="loss")
    fig3.savefig(loss_save_path)

    #print(action_abs)
    #print(f_reward_comparison)
    fig4, ax4_1 = plt.subplots()
    x_ax4 = np.linspace(0, epochs, epochs)
    labels = ["action_abs", "reward_comparison", "q_value"]
    ax4_2 = ax4_1.twinx()
    ax4_3 = ax4_1.twinx()
    ax4_1.plot(x_ax4, action_abs, color="r", label="action_abs")
    ax4_1.set_xlabel("epoch")
    ax4_1.set_ylabel("action_abs")
    ax4_2.plot(x_ax4, f_reward_comparison, color="b", label="reward_comparison")
    ax4_2.set_ylabel("reward_comparison")
    ax4_3.plot(x_ax4, qvalue_stack, color="g", label="q_value")
    ax4_3.set_ylabel("q_value")
    ax4_3.spines["right"].set_position(("axes", 1.15))
    handler1, label1 = ax4_1.get_legend_handles_labels()
    handler2, label2 = ax4_2.get_legend_handles_labels()
    handler3, label3 = ax4_3.get_legend_handles_labels()
    ax4_1.legend(handler1 + handler2 + handler3, label1 + label2 + label3, loc='upper center', bbox_to_anchor=(.5, -.15), ncol=3)
    fig4.savefig(arq_save_path,bbox_inches='tight')

    fig5 = plt.figure(figsize=(12, 8))
    ax2 = fig5.add_subplot(111)
    plt.title(fig_title)
    plt.plot(x_ax1, action_abs, label="action_abs")
    fig5.savefig(action_abs_save_path)

    fig6 = plt.figure(figsize=(12, 8))
    ax2 = fig6.add_subplot(111)
    plt.title(fig_title)
    plt.plot(x_ax1, qvalue_stack, label="q_value")
    fig6.savefig(qvalue_save_path)

    fig7 = plt.figure(figsize=(12, 8))
    ax2 = fig7.add_subplot(111)
    plt.title(fig_title)
    plt.plot(x_ax1, val_reward_stack, label="val_reward")
    fig7.savefig(val_reward_save_path)

def feature_norm(videos_src):    
    videos_src = videos_src.to(device)
    videos_src2 = videos_src.reshape((videos_src.shape[1], videos_src.shape[2]))
    videos = videos_src2 / torch.sqrt(torch.unsqueeze(torch.sum(videos_src2**2, axis=1),1))
    sims = torch.mm(videos, videos.t())
    sims.fill_diagonal_(-100)
    sim_max = sims.max()
    sims.fill_diagonal_(100)
    sim_min = sims.min()
    videos = torch.unsqueeze(videos, 0)
    return videos, sim_max, sim_min

#行動を選択する．
def select_action(state, frame):
    global steps_done
    EPS_START = 0.9
    EPS_END = 0.05#0.1#0.05
    EPS_DECAY = 500#200

    if frame == 0:  #初めのフレームの時
        return first_action
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).argmax().view(1)
    else:
        return torch.tensor([random.randrange(n_actions)], dtype=torch.long).to(device)

def optimize_model():
    BATCH_SIZE = 128
    GAMMA = 0.7#0.999#0.9#0.8#0.7#0.6#0.5
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool).to(device)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    #df.set_trace()

    next_state_values = torch.zeros(BATCH_SIZE).to(device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()  #逆伝播

    for param in policy_net.parameters():
        #df.set_trace()
        param.grad.data.clamp_(-1, 1)
        #df.set_trace()
    optimizer.step()
    return loss


#@profile
def main(train_video_name, val_video_name):

    csv_path = "./oneVideo_result/" + "train_" + train_video_name + "-val_" + val_video_name + "/csv_file/"
    if not os.path.isdir(csv_path):
        os.makedirs(csv_path)
    csv_file1 = os.path.join(csv_path, "all_result.csv")
    csv_file2 = os.path.join(csv_path, "recode_result_comparison.csv")
    csv_file3 = os.path.join(csv_path, "q_value_action.csv")
    csv_file4 = os.path.join(csv_path, "qaverage_actionabs.csv")
    csv_file5 = os.path.join(csv_path, "reward_comparison_epochaverage.csv")

    #それぞれのcsvファイルに何の数字が記録されているかを記す．
    with open(csv_file1, "w") as f1:
        writer = csv.writer(f1)
        data = ["Epoch","train_Video","Frame","Action","Reward","Reward_comparison","Loss"]
        writer.writerow(data)

    with open(csv_file2, "w") as f2:
        writer = csv.writer(f2)
        data = ["train_Video","Reinforce_min","Reinforce_ave","Reinforce_max"]
        writer.writerow(data)

    with open(csv_file3, "w") as f3:
        writer = csv.writer(f3)
        data = ["Epoch", "val_Video", "Frame", "Action_or_abs", "Q_value_or_average", "Reward"]
        writer.writerow(data)

    with open(csv_file4, "w") as f4:
        writer = csv.writer(f4)
        data = ["Epoch", "val_Video", "Action_abs", "Q_value_average", "Reward_average"]
        writer.writerow(data)

    with open(csv_file5, "w") as f5:
        writer = csv.writer(f5)
        data = ["Epoch", "train_Video", "Reward_comparison"]
        writer.writerow(data)

    set_seed()
    n_videos = 1  #使用する動画の本数
    epochs = 5000#500  #エポック数
    recode_epoch = epochs-30  #記録するエポックは後ろから30個だけ
    TARGET_UPDATE = 1000
    train_range = (0, 1)

    e_max_reward = 0
    recode_reward = 0
    recode_min_reward = 100
    recode_max_reward = 0
    recode_ave_reward = 0

    f_reward = []
    f_reward_comparison = []
    f_loss = []
    n_loss = 0
    action_abs = []
    qvalue_stack = []
    val_reward_stack = []

    """

    if video_name == "BG_10241":
        dfs_actions = [10,10,10,10,10,1,4,8]
    if video_name == "BG_10864":
        dfs_actions = [10,5,3,10,10,8,3,3]
    if video_name == "BG_11369":
        dfs_actions = [10,10,9,4,1,5,1,0]
    if video_name == "BG_12460":
        dfs_actions = [10,10,9,10,4,3,1,0]
    if video_name == "BG_13683":
        dfs_actions = [10,4,10,7,7,6,10,10]
    else:
        dfs_actions = [10,10,10,10,10,1,4,8]

    """

    dfs_all_csv_path = "dfs_result/dfs_result_max_action.csv"
    dfs_actions = []
    with open(dfs_all_csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for low in reader:
            if val_video_name == low[0]:
                for i in range(len(low)):
                    if i == 0:
                        continue
                    dfs_actions.append(int(low[i]))
    

    for epoch in range(epochs):
        print("Epoch : {}/{}".format(epoch, epochs))
        n_allvid_frame = 0
        e_reward = 0
        e_reward_comparison = 0
        csv_data = []
        e_loss = 0
        for feats_path in glob.glob("./Features/AllKeyVideos/" + train_video_name + "_features.h5"):
            feature = get_eval_loader(train_range, feats_path)
            #print(feats_path)
            for v_num, (videos_src, video_ids) in enumerate(feature):
                videos, sim_max, sim_min = feature_norm(videos_src)
                max_frame = videos.shape[1]
                n_allvid_frame += max_frame
                V_p = env.reset(sim_max, sim_min)
                #action_stack = []
                for frame in range(max_frame-1):
                    v_t = videos[0, frame]
                    state = torch.unsqueeze(torch.cat((V_p, v_t), dim = 0), 0)
                    action = select_action(state, frame)
                    v_next = videos[0, frame+1]
                    V_p, reward, reward_comparison = env.step(action, v_t, v_next)
                    next_state = torch.unsqueeze(torch.cat((V_p, v_next), dim = 0), 0)
                    action = torch.unsqueeze(action,0)
                    memory.push(state, action, next_state, reward)
                    loss = optimize_model()

                    if loss is not None:
                        n_loss += 1
                        loss = loss.item()
                        f_loss.append(loss)
                    data = [str(epoch),feats_path.split("/")[3].split("_")[1],str(frame),str(action.item()),str(reward.item()),str(reward_comparison.item()), str(loss)]
                    csv_data.append(data)

                    e_reward += reward
                    e_reward_comparison += reward_comparison
                    #action_stack.append(action.item())
                    
                    # 深さ優先探索との比較で使用する(break)
                    if frame == 7:
                        n_allvid_frame = 9
                        break

                # 各映像ごとの報酬を計算するならこの位置に
                #action_abs.append(np.sum(np.abs([x - y for (x, y) in zip(dfs_actions, action_stack)])))
        # エピソード毎の報酬を計算するならこの位置に
        e_reward = e_reward / (n_allvid_frame - n_videos)
        e_reward_comparison = e_reward_comparison / (n_allvid_frame - n_videos)
        f_reward.append(e_reward.item())
        f_reward_comparison.append(e_reward_comparison.item())
        e_qvalue,e_action,val_reward = compute_qvalue(val_video_name, dfs_actions, csv_file3, epoch, csv_file4)
        qvalue_stack.append(e_qvalue)
        action_abs.append(e_action)
        val_reward_stack.append(val_reward)

        with open(csv_file5, "a") as f5:
            writer = csv.writer(f5)
            data = [str(epoch), train_video_name, str(e_reward_comparison.item())]
            writer.writerow(data)


        with open(csv_file1, "a") as f1:
            writer = csv.writer(f1)
            writer.writerows(csv_data)

        if (epoch+1) % 500 == 0:
            model_save(train_video_name, str(epoch+1), val_video_name)

        if e_max_reward < e_reward_comparison:
            e_max_reward = e_reward_comparison
            model_save(train_video_name, "max", val_video_name)
        if (epoch+1) % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        if recode_epoch <= epoch:
            recode_reward += e_reward_comparison
            if recode_min_reward > e_reward_comparison:
                recode_min_reward = e_reward_comparison
            if recode_max_reward < e_reward_comparison:
                recode_max_reward = e_reward_comparison
    recode_ave_reward = recode_reward / (epochs - recode_epoch)

    with open(csv_file2, "a") as f2:
        data = [train_video_name, str(recode_min_reward.item()), str(recode_ave_reward.item()), str(recode_max_reward.item())]
        writer = csv.writer(f2)
        writer.writerow(data)

    print("create fig")
    fig_creat(epochs,n_loss,f_loss,f_reward,f_reward_comparison,train_video_name,action_abs,qvalue_stack,val_reward_stack,val_video_name)
    print("Fin!!")

if __name__ == "__main__":
    args = sys.argv
    train_video_name = args[1]
    val_video_name = args[2]
    main(train_video_name, val_video_name)
