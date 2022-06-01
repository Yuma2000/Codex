# --- Import section --- #
import random
from collections import namedtuple
# --- Variable declaration section --- #
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
# --- Code section --- #
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        #if len(self.memory) > 20:
        #    self.memory = []
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
