import numpy as np
import random
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
FeaturesTransition = namedtuple("FeaturesTransition", ('features', 'action', 'next_features', 'reward'))
EpisodesStats = namedtuple("Stats", ('rewards'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """SAVE A Transition. """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = FeaturesTransition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size=None):
        if batch_size == None:
            return self.memory
        else:
            return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def reset(self):
        self.memory = []
        self.position = 0
