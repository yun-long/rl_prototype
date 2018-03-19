import numpy as np
from rl.featurizer.one_hot_featurizer import OneHotFeaturizer
from gym.spaces.discrete import Discrete
from gym.spaces.tuple_space import Tuple
from collections import defaultdict
import itertools
#


class RandomPolicy(object):
    def __init__(self, env):
        self.nA = env.action_space.n
        self.action_probs = np.ones(self.nA, dtype=float) / self.nA

    def predict_action(self, state):
        action = np.random.choice(np.arange(len(self.action_probs)),p=self.action_probs)
        return action

class DistributionPolicy_V1(object):
    """
    The policy is defined as the distribution of state-action pairs.
    Specifically for env.action_space == Tuple(Discrete, Discrete)
    """
    def __init__(self, env, rnd=None):
        assert isinstance(env.action_space, Tuple)
        assert isinstance(env.observation_space, Discrete)
        if rnd is not None:
            self.rnd = rnd
        self.env = env
        self.obs_dim = env.observation_space.n
        action_tmp = []
        for i, space in enumerate(env.action_space.spaces):
            action_tmp.append([])
            for j in range(space.n):
                action_tmp[i].append(j)
        self.actions = list(itertools.product(*action_tmp))
        self.act_dim = len(self.actions)
        self.q_dist = np.ones(shape=(self.obs_dim, self.act_dim)) / (self.obs_dim * self.act_dim)
        self.mu_dist = np.ones(self.obs_dim) / self.obs_dim
        self.pi = self.q_dist / self.mu_dist[:, None]

    def predict_action(self, state):
        action_prob = self.pi[state]
        action_idex = self.rnd.choice(np.arange(len(action_prob)), p=action_prob)
        action = self.actions[action_idex]
        # print(action)
        return action_idex, action

    def update_reps(self, A, param_eta, param_v, g, keys):
        adv_sa = g * np.ones((self.obs_dim, self.act_dim))
        adv_sa[tuple(zip(*keys))] = A(param_v)
        weights = np.exp((adv_sa-np.max(adv_sa))/param_eta)
        pi_new = np.copy(self.pi)
        pi_new *= weights
        pi_new /= np.sum(pi_new, axis=1, keepdims=True)
        self.pi = pi_new

    def update_freps(self, A, eta, param_lamda, param_v, keys, fcp, param_kappa=None):
        adv_sa = param_lamda * np.ones((self.obs_dim, self.act_dim))
        adv_sa[tuple(zip(*keys))] = A(param_v)
        if param_kappa is not None:
            kappa = np.zeros((self.obs_dim, self.act_dim))
            kappa[tuple(zip(*keys))] = param_kappa
            y = (adv_sa - param_lamda + kappa) / eta
        else:
            y = (adv_sa - param_lamda) / eta
        weights = fcp(y)
        pi_new = np.copy(self.pi)
        pi_new *= weights
        pi_new /= np.sum(pi_new, axis=1, keepdims=True)
        self.pi = pi_new

class DistributionPolicy(object):
    """
    The policy is defined as the distribution of state-action pairs.
    Specifically for env.action_space == Discrete
    """
    def __init__(self, env, rnd=None):
        assert isinstance(env.action_space, Discrete)
        if rnd is not None:
            self.rnd = rnd
        self.env = env
        # state action distribution
        self.q_dist = np.ones((env.observation_space.n, env.action_space.n)) \
                      / (env.observation_space.n * env.action_space.n)
        # state distribution
        self.mu_dist = np.ones(env.observation_space.n) / env.observation_space.n
        # policy
        self.pi = self.q_dist / self.mu_dist[:, None]

    def predict_action(self, state):
        action_prob = self.pi[state]
        action = self.rnd.choice(np.arange(len(action_prob)), p=action_prob)
        return action

    def update_reps(self, A, param_eta, param_v, g, keys):
        adv_sa = g * np.ones((self.env.observation_space.n, self.env.action_space.n))
        adv_sa[tuple(zip(*keys))] = A(param_v)
        weights = np.exp((adv_sa-np.max(adv_sa))/param_eta)
        pi_new = np.copy(self.pi)
        pi_new *= weights
        pi_new /= np.sum(pi_new, axis=1, keepdims=True)
        self.pi = pi_new

    def update_freps(self, A, eta, param_lamda, param_v, keys, fcp, param_kappa=None):
        adv_sa = param_lamda * np.ones((self.env.observation_space.n, self.env.action_space.n))
        adv_sa[tuple(zip(*keys))] = A(param_v)
        if param_kappa is not None:
            kappa = np.zeros((self.env.observation_space.n, self.env.action_space.n))
            kappa[tuple(zip(*keys))] = param_kappa
            y = (adv_sa - param_lamda + kappa) / eta
        else:
            y = (adv_sa - param_lamda) / eta
        #
        weights = fcp(y)
        pi_new = np.copy(self.pi)
        pi_new *= weights
        pi_new /= np.sum(pi_new, axis=1, keepdims=True)
        self.pi = pi_new

class GreedyPolicy(object):
    def __init__(self, env, Q):
        self.nA = env.action_space.n
        self.Q = Q

    def predict_action(self, state):
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

