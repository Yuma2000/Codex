# --- Import section --- #
import torch
import torch.nn as nn

# --- Variable declaration section --- #

# --- Code section --- #
class DQN(nn.Module):
    def __init__(self, V_p, V_t, outputs=1):
        super(DQN, self).__init__()
        self.input = nn.Linear(V_p + V_t, outputs)

    def forward(self, x):
        action = self.input(x)
        return action
