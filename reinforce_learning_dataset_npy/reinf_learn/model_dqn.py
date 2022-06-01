# --- Import section --- #
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Variable declaration section --- #

# --- Code section --- #
class DQN(nn.Module):
    def __init__(self, V_p, V_t, outputs=1):
        super(DQN, self).__init__()

        # self.input = nn.Linear(V_p + V_t, outputs)
        self.inp_shape = V_p + V_t
        #self.l1 = nn.Linear(self.inp_shape, 4096)
        #self.bn1 = nn.BatchNorm1d(4096)
        # self.bn2 = nn.BatchNorm1d(1024)
        self.bn1 = nn.LayerNorm(4096)
        #self.l2 = nn.Linear(4096, 1024)
        # self.bn2 = nn.BatchNorm1d(1024)
        #self.bn2 = nn.LayerNorm(4096)
        self.out = nn.Linear(1024, outputs)
        self.l1 = nn.Linear(self.inp_shape, 2048)
        # self.bn1 = nn.BatchNorm1d(2048)
        #self.bn1 = nn.LayerNorm(2048)
        #self.bn2 = nn.LayerNorm(2048)
        self.l2 = nn.Linear(2048, 1024)
        # self.out = nn.Linear(1024, outputs)
        #self.l1 = nn.Linear(self.inp_shape, 1024)
        # self.bn1 = nn.BatchNorm1d(1024)
        #self.bn1 = nn.LayerNorm(1024)
        #self.bn2 = nn.LayerNorm(1024)
        #self.l2 = nn.Linear(1024, 1024)
        # self.out = nn.Linear(1024, outputs)

    def forward(self, x):
        # action = self.input(x)
        # x = F.relu(self.input(x))
        #x = F.relu(self.bn1(self.l1(x)))
        x = F.relu(self.l1(self.bn1(x)))
        #x = F.relu(self.l1(x))
        #x = F.relu(self.bn2(self.l2(x)))
        #x = F.relu(self.l2(self.bn2(x)))
        x = F.relu(self.l2(x))
        #x = self.relu(x)
        action = self.out(x)
        return action
