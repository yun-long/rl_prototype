import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition', ['state', 'action', 'next_state', 'reward'])
EpisodesStats = namedtuple("Stats", ['rewards'])

def discount_norm_rewards(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)
    return  discounted_rewards


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """SAVE A Transition. """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size=None):
        if batch_size == None:
            return self.memory
        else:
            return np.random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
