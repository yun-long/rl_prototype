import numpy as np
#
from gym.spaces.tuple_space import Tuple
from gym.spaces.discrete import Discrete
from rl.misc.memory import Transition
from collections import defaultdict
#
class TestSampler(object):

    def __init__(self, env):
        if isinstance(env.action_space, Tuple):
            self.act_type = 'tuple'
        elif isinstance(env.action_space, Discrete):
            self.act_type = 'discrete'
        else:
            raise NotImplementedError
        self.env = env
        self.keys = ['state', 'action', 'reward', 'done']

    def create_path(self):
        paths = {}
        for key in self.keys:
            paths[key] = []
        n_path = 0
        return paths, n_path

    def rollout(self, policy):
        state = self.env.reset()
        done = False
        while not done:
            action = policy(state)
            next_state, reward, done, _ = self.env.step(action)
            yield state, action, reward, done
            state = next_state

    def rollouts(self, policy, n_timesteps):
        paths, n_path = self.create_path()
        while len(paths['reward']) < n_timesteps:
            for transition in self.rollout(policy):
                for key, tran in zip(self.keys, transition):
                    paths[key].append(tran)
                n_path += 1
        for key in self.keys:
            paths[key] = np.asarray(paths[key])
            if paths[key].ndim == 1:
                paths[key] = np.expand_dims(paths[key], axis=-1)
        paths['n_path'] = n_path
        return paths

    def get_adv(self, paths, val_fn, featurizer, discount, lam):
        n, r, features_diff = defaultdict(float), defaultdict(float), defaultdict(float)
        advantages = np.empty_like(paths['reward'])
        for rev_t, val in enumerate(reversed(paths['reward'])):
            t = len(paths['reward']) - rev_t - 1
            state_action = (paths['state'][t], paths['action'])
            n[state_action] += 1
            features_diff[state_action] += featurizer.transform(paths['state'][t+1]) - featurizer.transform(paths['state'][t])
            if paths['done'][t]:
                advantages[t] = paths['reward'][t] - val_fn(paths['state'][t])
            else:
                sigma = paths['reward'][t] + discount * val_fn(paths['state'][t+1]) - val_fn(paths['state'][t])
                advantages[t] = sigma + lam * advantages[t+1]





