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
from model import DQN, DQN_svr, KFrame_memory_DQN
from replayMemory import Transition
from replayMemory import ReplayMemory
from dataLoader import get_eval_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
torch.cuda.manual_seed(0)
now = datetime.datetime.now()
env = gym.make("KFrame_Memory-v0")
env.reset(1, -1)
n_actions = env.action_space.n
##
action_num = 11#6
k_size = action_num -1
##
policy_net = KFrame_memory_DQN(2048*k_size, 2048, n_actions).to(device)
target_net = KFrame_memory_DQN(2048*k_size, 2048, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr = 1e-6, weight_decay=0)
steps_done = 0
memory_size = 10000
memory = ReplayMemory(memory_size)
first_action = torch.tensor([n_actions-1], dtype=torch.long).to(device)

def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def compute_qvalue(dfs_actions, epoch, csv_file3, csv_file4, val_csv_paths):
    ratio_list = torch.from_numpy(np.array([[i*0.1] for i in range(11)])).float().to(device)
    ratio_list[0] = 1e-7

    for feats_path in val_csv_paths:
        train_range = (0,1)
        action = []
        q_value = []
        q_stack = 0.0
        action_stack = []
        reward_pool = []
        feature = get_eval_loader(train_range, feats_path)
        val_video_name = feats_path.split("/")[3].split("_")[0] + "_" + feats_path.split("/")[3].split("_")[1]
        dfs_all_csv_path = "./dfs_result/dfs_result_max_action.csv"
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
        for v_num, (videos_src, video_ids) in enumerate(feature):
            videos, sim_max, sim_min = feature_norm(videos_src)
            max_frame = videos.shape[1]
            #V_p = torch.zeros(2048, dtype=torch.float).to(device)
            V_p = torch.zeros(k_size,2048, dtype=torch.float).to(device)
            for frame in range(max_frame-1):
                v_t = videos[0, frame]
                v_next = videos[0, frame+1]
                #state = torch.unsqueeze(torch.cat((V_p, v_t), dim = 0), 0)
                state = torch.unsqueeze(torch.cat((torch.flatten(V_p), v_t), dim = 0), 0)
                #if frame == 0:
                    ##q_stack = target_net(state)[0][n_actions-1].item()
                    #q_stack = policy_net(state)[0][n_actions-1].item()
                    #action = n_actions-1
                if frame < k_size:
                    q_stack = policy_net(state)[0][frame].item()
                    action = frame
                else:
                    #q_stack = target_net(state).max(1)[0].item()
                    #action = target_net(state).max(1)[1].item()
                    q_stack = policy_net(state).max(1)[0].item()
                    action = policy_net(state).max(1)[1].item()
                    
                #mem = V_p + (torch.repeat_interleave(v_t, 11).reshape(2048,11).T)*ratio_list
                #mem = mem / (torch.norm(mem, dim=1).unsqueeze(1))
                #V_p = mem[action].squeeze(0)
                #mem_2 = torch.mv(mem, v_next.T)
                #r_norm = ((mem_2[action] - sim_min) / (sim_max - sim_min)).item()

                if action < k_size:
                    V_p[action] = v_t
                #mem_list = torch.Tensor([torch.dot(V_p[i],v_next.T) for i in range(k_size)]).float().to(device)
                mem_list = torch.mv(V_p,v_next.T)
                mem = torch.max(mem_list)
                r_norm = ((mem - sim_min) / (sim_max - sim_min)).item()

                q_value.append(q_stack)
                action_stack.append(action)
                reward_pool.append(r_norm)

                with open(csv_file3, "a") as f3:
                    data = [str(epoch), val_video_name, str(frame), str(action), str(q_stack), str(r_norm)]
                    writer = csv.writer(f3)
                    writer.writerow(data)
                
                #if frame == 7:
                    #break
        #print(q_value)
        e_action = np.sum(np.abs([x - y for (x, y) in zip(dfs_actions, action_stack)]))
        qvalue_average = np.mean(q_value)
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


def model_save(train_video_name, key, val_video_name):
    # m_save_path = "./kFrame_Memory_result/Model_save/"
    m_save_path = "./kFrame_Memory_result_for_onevideo/" + "train_" + train_video_name + "-val_" + val_video_name + "/Model_save/"
    if not os.path.isdir(m_save_path):
        os.makedirs(m_save_path)
    target_path = "target_" + key + ".pth"
    policy_path = "policy_" + key + ".pth"
    model_t_path = os.path.join(m_save_path, target_path)
    model_p_path = os.path.join(m_save_path, policy_path)
    torch.save(policy_net.state_dict(), model_p_path)
    torch.save(target_net.state_dict(), model_t_path)


def fig_creat(epochs,n_loss,f_loss,f_reward,f_reward_comparison, video_name,action_abs,qvalue_stack,val_reward_stack,val_video_name):
    #f_save_path = "./kFrame_Memory_result/Fig_save/"
    f_save_path = "./kFrame_Memory_result_for_onevideo/" + "train_" + train_video_name + "-val_" + val_video_name + "/Fig_save/"
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

def select_action(state, frame):
    global steps_done
    EPS_START = 0.9
    EPS_END = 0.05#0.1#0.05
    EPS_DECAY = 500#200

    #if frame == 0:
        #return first_action
    if frame < k_size:
        return torch.tensor([frame], dtype=torch.long).to(device)
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
    GAMMA = 0.5#0.7#0.5
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

    next_state_values = torch.zeros(BATCH_SIZE).to(device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss

#@profile
def main(train_video_name, val_video_name):

    
    # csv_path = "./kFrame_Memory_result/csv_file/"
    csv_path = "./kFrame_Memory_result_for_onevideo/" + "train_" + train_video_name + "-val_" + val_video_name + "/csv_file/"
    if not os.path.isdir(csv_path):
        os.makedirs(csv_path)
    csv_file1 = os.path.join(csv_path, "all_result.csv")
    csv_file2 = os.path.join(csv_path, "recode_result_comparison.csv")
    csv_file3 = os.path.join(csv_path, "q_value_action.csv")
    csv_file4 = os.path.join(csv_path, "qaverage_actionabs.csv")
    csv_file5 = os.path.join(csv_path, "reward_comparison_epochaverage.csv")
    csv_file6 = os.path.join(csv_path, "reward_average_epochs.csv")

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
        data = ["Epoch", "val_Video", "Frame", "Action_or_8frameABS", "Q_value_or_allframe_average", "Reward_or_allframe_average"]
        writer.writerow(data)

    with open(csv_file4, "w") as f4:
        writer = csv.writer(f4)
        data = ["Epoch", "val_Video", "8frame_Action_abs", "allframe_Q_value_average", "allframe_Reward_average"]
        writer.writerow(data)

    with open(csv_file5, "w") as f5:
        writer = csv.writer(f5)
        data = ["Epoch", "train_Video", "Reward_comparison"]
        writer.writerow(data)
    
    with open(csv_file6, "w") as f6:
        writer = csv.writer(f6)
        data = ["Epoch", "train_Video", "Reward_average", "Reward_comparison_average"]
        writer.writerow(data)

    set_seed()
    n_videos = 1#219
    epochs = 10000#5000
    recode_epoch = epochs-30#epochs-3#epochs-30
    TARGET_UPDATE = 1000#10#1#10#1000
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
    dfs_actions = []
    val_reward_stack = []

    reward_ave_epoch = []
    
    keys = lambda x: (x.split("/")[3].split("_")[1])
    #train_csv_paths = sorted(glob.glob("./Features/AllKeyVideos/*_features.h5"), key = keys)
    #val_csv_paths = sorted(glob.glob("./Features/AllKeyVideos/*_features.h5"), key = keys)
    train_csv_paths = ["./Features/AllKeyVideos/"+train_video_name+"_features.h5"]
    val_csv_paths = ["./Features/AllKeyVideos/"+val_video_name+"_features.h5"]

    for epoch in range(epochs):
        print("Epoch : {}/{}".format(epoch, epochs))
        fivevid_frame = 0
        n_allvid_frame = 0
        e_reward = 0
        e_reward_comparison = 0
        csv_data = []
        e_loss = 0

        for feats_path in train_csv_paths:
            feature = get_eval_loader(train_range, feats_path)
            video_name = feats_path.split("/")[3].split("_")[0] + "_" + feats_path.split("/")[3].split("_")[1]

            for v_num, (videos_src, video_ids) in enumerate(feature):
                videos, sim_max, sim_min = feature_norm(videos_src)
                max_frame = videos.shape[1]
                n_allvid_frame += max_frame
                reward_epochs = []
                reward_comparison_epochs = []
                V_p = env.reset(sim_max, sim_min)
                for frame in range(max_frame-1):
                    v_t = videos[0, frame]

                    #state = torch.unsqueeze(torch.cat((V_p, v_t), dim = 0), 0)
                    #print(V_p.shape)
                    #print(torch.flatten(V_p).shape)
                    state = torch.unsqueeze(torch.cat((torch.flatten(V_p), v_t), dim = 0), 0)
                    action = select_action(state, frame)
                    v_next = videos[0, frame+1]
                    #print(action)
                    V_p, reward, reward_comparison = env.step(action, v_t, v_next)
                    #next_state = torch.unsqueeze(torch.cat((V_p, v_next), dim = 0), 0)
                    next_state = torch.unsqueeze(torch.cat((torch.flatten(V_p), v_next), dim = 0), 0)
                    action = torch.unsqueeze(action,0)
                    #reward = torch.unsqueeze(torch.unsqueeze(reward,0),0)
                    #print(action)
                    #print(reward)
                    reward = torch.unsqueeze(reward,0)
                    memory.push(state, action, next_state, reward)
                    loss = optimize_model()

                    if loss is not None:
                        n_loss += 1
                        loss = loss.item()
                        f_loss.append(loss)
                    data = [str(epoch),video_name,str(frame),str(action.item()),str(reward.item()),str(reward_comparison.item()), str(loss)]
                    csv_data.append(data)

                    reward_epochs.append(reward.item())
                    reward_comparison_epochs.append(reward_comparison.item())
                    e_reward += reward
                    e_reward_comparison += reward_comparison
                    #action_stack.append(action.item())
                    
                    # 深さ優先探索との比較で使用する(break)
                    #if frame == 7:
                        #n_allvid_frame = 9*n_videos #9->8フレームでやるときはこの値 #5->5映像分で学習するからこの値
                        #break

                # 各映像ごとの報酬を計算するならこの位置に
                """
                with open(csv_file6, "a") as f6:
                    writer = csv.writer(f6)
                    data = [str(epoch), video_name, str(np.mean(reward_epochs)), str(np.mean(reward_comparison_epochs))]
                    reward_epochs = []
                    reward_comparison_epochs = []
                    writer.writerow(data)
                """
                reward_ave_epoch.append([str(epoch), video_name, str(np.mean(reward_epochs)), str(np.mean(reward_comparison_epochs))])
                reward_epochs = []
                reward_comparison_epochs = []
                #action_abs.append(np.sum(np.abs([x - y for (x, y) in zip(dfs_actions, action_stack)])))
        # エピソード毎の報酬を計算するならこの位置に
        e_reward = e_reward / (n_allvid_frame - n_videos)
        e_reward_comparison = e_reward_comparison / (n_allvid_frame - n_videos)
        f_reward.append(e_reward.item())
        f_reward_comparison.append(e_reward_comparison.item())
        e_qvalue,e_action,val_reward = compute_qvalue(dfs_actions, epoch, csv_file3, csv_file4, val_csv_paths)
        qvalue_stack.append(e_qvalue)
        action_abs.append(e_action)
        val_reward_stack.append(val_reward)

        
        with open(csv_file5, "a") as f5:
            writer = csv.writer(f5)
            data = [str(epoch), "allVideo", str(e_reward_comparison.item())]
            writer.writerow(data)


        with open(csv_file1, "a") as f1:
            writer = csv.writer(f1)
            writer.writerows(csv_data)
        

        if (epoch+1) % 500 == 0:
            model_save(video_name, str(epoch+1),val_video_name)

        if e_max_reward < e_reward_comparison:
            e_max_reward = e_reward_comparison
            model_save(video_name, "max",val_video_name)
        if (epoch+1) % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        if recode_epoch <= epoch:
            #print(e_reward_comparison)
            recode_reward += e_reward_comparison
            if recode_min_reward > e_reward_comparison:
                recode_min_reward = e_reward_comparison
            if recode_max_reward < e_reward_comparison:
                recode_max_reward = e_reward_comparison
    recode_ave_reward = recode_reward / (epochs - recode_epoch)

    
    with open(csv_file2, "a") as f2:
        #print(recode_min_reward,recode_ave_reward,recode_max_reward)
        data = ["allVideo", str(recode_min_reward.item()), str(recode_ave_reward.item()), str(recode_max_reward.item())]
        writer = csv.writer(f2)
        writer.writerow(data)

    with open(csv_file6, "a") as f6:
        writer = csv.writer(f6)
        writer.writerows(reward_ave_epoch)
    

    print("create fig")
    fig_creat(epochs,n_loss,f_loss,f_reward,f_reward_comparison,video_name,action_abs,qvalue_stack,val_reward_stack,val_video_name)
    print("Fin!!")

if __name__ == "__main__":
    args = sys.argv
    train_video_name = args[1]
    val_video_name = args[2]
    main(train_video_name, val_video_name)


