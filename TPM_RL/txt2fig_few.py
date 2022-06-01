import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# vid7
epochs, videos, frames, actions, rewards = np.loadtxt("./log_text/20200128_105830.txt", delimiter=',', unpack=True, skiprows=1)
# vid0
#epochs, videos, frames, actions, rewards = np.loadtxt("./log_text/20200128_111540.txt", delimiter=',', unpack=True, skiprows=1)
#epochs, videos, frames, actions, rewards = np.loadtxt("./log_text/20200128_113407.txt", delimiter=',', unpack=True, skiprows=1)
#epochs, videos, frames, actions, rewards = np.loadtxt("./log_text/20200128_113336.txt", delimiter=',', unpack=True, skiprows=1)
#print(actions)20200128_113336

#act = []
#for i in range(len(epochs)):
#    if videos[i] == 0:
#        if frames[i] == 3:
#            act.append(actions[i])

#print(act)

#kep = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
#kfig = plt.figure(figsize=(20,10))
#kax = kfig.add_subplot(111)
#kax.plot(kep, act, "o-", color="b", label="transition of frame0")
#kax.set_xlim(-0.5, 30.5)
#kax.set_ylim(-0.05, 10.05)
#kax.set_xlabel("epoch")
#kax.set_ylabel("重み")
#kax.legend(loc="upper left")
#plt.show()

#print("Fin")

#epoch = []
video = []
frame0 = []
frame1 = []
frame2 = []
frame3 = []
frame4 = []
frame5 = []
frame6 = []
frame7 = []
frame8 = []
frame9 = []
frame10 = []
frame11 = []
frame12 = []
frame13 = []
frame14 = []
frame15 = []
frame16 = []
frame17 = []
frame18 = []
frame19 = []

action0 = []
action1 = []
action2 = []
action3 = []
action4 = []
action5 = []
action6 = []
action7 = []
action8 = []
action9 = []
action10 = []
action11 = []
action12 = []
action13 = []
action14 = []
action15 = []
action16 = []
action17 = []
action18 = []
action19 = []

action = []
reward = []

#ep = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
#ep = np.arange(0, 30, 1)
ep = np.arange(0, 300, 10)

fig1 = plt.figure(figsize=(15,9))
ax1 = fig1.add_subplot(111)
fig2 = plt.figure(figsize=(15,9))
ax2 = fig2.add_subplot(111)
#fig3 = plt.figure(figsize=(15,9))
#ax3 = fig3.add_subplot(111)
#fig4 = plt.figure(figsize=(15,9))
#ax4 = fig4.add_subplot(111)

#f = open("./log_text/20200109_151635.txt")
#txt = f.readline()

for i in range(len(epochs)):
    if epochs[i]%10 == 0:
    #for epochs, videos, frames, actions, rewards in txt:
        if videos[i] == 6:
        #if videos[i] == 0:
            #epoch.append(epochs)
            if frames[i] == 0:
                action0.append(actions[i])
            if frames[i] == 1:
                action1.append(actions[i])
            if frames[i] == 2:
                action2.append(actions[i])
            if frames[i] == 3:
                action3.append(actions[i])
            if frames[i] == 4:
                action4.append(actions[i])
            if frames[i] == 5:
                action5.append(actions[i])
            if frames[i] == 6:
                action6.append(actions[i])
            if frames[i] == 7:
                action7.append(actions[i])
            if frames[i] == 8:
                action8.append(actions[i])
            if frames[i] == 9:
                action9.append(actions[i])
            if frames[i] == 10:
                action10.append(actions[i])
            if frames[i] == 11:
                action11.append(actions[i])
            if frames[i] == 12:
                action12.append(actions[i])
            if frames[i] == 13:
                action13.append(actions[i])
            if frames[i] == 14:
                action14.append(actions[i])
            if frames[i] == 15:
                action15.append(actions[i])
            if frames[i] == 16:
                action16.append(actions[i])
            if frames[i] == 17:
                action17.append(actions[i])
            if frames[i] == 18:
                action18.append(actions[i])
            if frames[i] == 19:
                action19.append(actions[i])


ax1.plot(ep, action9, "o-", color="b", label="frame9")
ax1.plot(ep, action11, "o-", color="r", label="frame11")
#ax1.plot(ep, action2, "o-", color="r", label="transition of frame2")
#ax1.plot(ep, action3, "o-", color="c", label="transition of frame3")
#ax1.plot(ep, action4, "o-", color="m", label="transition of frame4")

ax2.plot(ep, action8, "o-", color="b", label="frame8")
ax2.plot(ep, action17, "o-", color="r", label="frame17")
#ax2.plot(ep, action7, "o-", color="r", label="transition of frame7")
#ax2.plot(ep, action8, "o-", color="c", label="transition of frame8")
#ax2.plot(ep, action9, "o-", color="m", label="transition of frame9")

ax1.set_xlim(-0.5, 300.5)
ax1.set_ylim(-0.5, 10.5)
ax2.set_xlim(-0.5, 300.5)
ax2.set_ylim(-0.5, 10.5)

ax1.set_xlabel("epoch",fontsize=30)
ax1.set_ylabel("Action",fontsize=30)
ax2.set_xlabel("epoch",fontsize=30)
ax2.set_ylabel("Action",fontsize=30)

#ax1.legend(loc="lower right",fontsize=20)
#ax2.legend(loc="upper right",fontsize=20)

ax1.legend(loc="upper center", ncol=5, bbox_to_anchor=(0.5,1.15), fontsize=30)
ax2.legend(loc="upper center", ncol=5, bbox_to_anchor=(0.5,1.15), fontsize=30)

ax1.tick_params(labelsize=30)
ax2.tick_params(labelsize=30)

plt.show()

plt.savefig('figure1.png')
plt.savefig('figure2.png')
#plt.savefig('figure3.png')
#plt.savefig('figure4.png')

#f.close()
