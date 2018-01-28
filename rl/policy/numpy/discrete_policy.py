import numpy as np
#

class RandomPolicy(object):
    def __init__(self, env):
        self.nA = env.action_space.n
        self.action_probs = np.ones(self.nA, dtype=float) / self.nA

    def predict(self, state):
        action = np.random.choice(np.arange(len(self.action_probs)),p=self.action_probs)
        return action


class GreedyPolicy(object):
    def __init__(self, env, Q):
        self.nA = env.action_space.n
        self.Q = Q

    def predict(self, state):
        A_probs = np.zeros_like(self.Q[state], dtype=float)
        best_action = np.argmax(self.Q[state])
        A_probs[best_action] = 1.0
        action = np.random.choice(np.arange(len(A_probs)), p=A_probs)
        return action

    def update(self, Q):
        self.Q = Q


class EpsilonGreedyPolicy(object):

    def __init__(self, env, Q, epsilon):
        self.nA = env.action_space.n
        self.Q = Q
        self.epsilon = epsilon

    def predict(self, state):
        A = np.ones(self.nA, dtype=float) * self.epsilon / self.nA
        best_action = np.argmax(self.Q[state])
        A[best_action] += (1.0 - self.epsilon)
        action = np.random.choice(np.arange(len(A)), p=A)
        return action

    def update(self, Q):
        self.Q = Q