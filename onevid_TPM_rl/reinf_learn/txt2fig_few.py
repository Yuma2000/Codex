import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# vid7
#epochs, videos, frames, actions, rewards = np.loadtxt("./log_text/20200227_114611.txt", delimiter=',', unpack=True, skiprows=1)
# vid0
epochs, videos, frames, actions, rewards = np.loadtxt("./log_text/20200227_163755.txt", delimiter=',', unpack=True, skiprows=1)
#epochs, videos, frames, actions, rewards = np.loadtxt("./log_text/20200128_113407.txt", delimiter=',', unpack=True, skiprows=1)
#epochs, videos, frames, actions, rewards = np.loadtxt("./log_text/20200128_113336.txt", delimiter=',', unpack=True, skiprows=1)

foa = [[] for b in range(99)]
for i in range(len(epochs)):
    #if i % 10 == 0 and videos[i] == 0:
    if epochs[i] % 10 == 0 and videos[i] == 0:
    #if videos[i] == 0:
        for j in range(99):
            if frames[i] == j:
                foa[int(frames[i])].append(actions[i])
                print(epochs[i],frames[i])

ep = np.arange(0, 3000, 10)
#ep = np.arange(0, 300, 1)

fig1 = plt.figure(figsize=(15,9))
ax1 = fig1.add_subplot(111)
fig2 = plt.figure(figsize=(15,9))
ax2 = fig2.add_subplot(111)

ax1.plot(ep, foa[0], "o-", color="b", label="frame")
ax1.plot(ep, foa[44], "o-", color="r", label="frame")

ax2.plot(ep, foa[60], "o-", color="b", label="frame")
ax2.plot(ep, foa[80], "o-", color="r", label="frame")

ax1.set_xlim(-0.5, 3000.5)
ax1.set_ylim(-0.5, 10.5)
ax2.set_xlim(-0.5, 3000.5)
ax2.set_ylim(-0.5, 10.5)

ax1.set_xlabel("epoch",fontsize=30)
ax1.set_ylabel("Action",fontsize=30)
ax2.set_xlabel("epoch",fontsize=30)
ax2.set_ylabel("Action",fontsize=30)

ax1.legend(loc="upper center", ncol=5, bbox_to_anchor=(0.5,1.15), fontsize=30)
ax2.legend(loc="upper center", ncol=5, bbox_to_anchor=(0.5,1.15), fontsize=30)

ax1.tick_params(labelsize=30)
ax2.tick_params(labelsize=30)

plt.show()

plt.savefig('figure1.png')
plt.savefig('figure2.png')