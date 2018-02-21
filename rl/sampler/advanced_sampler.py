import numpy as np

class AdvancedSampler(object):

    def __init__(self, env):
        self.env = env
        self.keys = ['state', 'action', 'reward', 'done']

    def create_path(self):
        self.paths = {}
        for key in self.keys:
            self.paths[key] = []
        self.n_path = 0
        self.n_transitions = 0

    def rollout(self, policy):
        state = self.env.reset()
        done = False
        while not done:
            self.n_transitions += 1
            action = policy(state)
            next_state, reward, done, _ = self.env.step(action)
            yield state, action, reward, done
            state = next_state

    def rollous(self, policy, n_trans):
        self.create_path()
        while len(self.paths) < n_trans:
            for transition in self.rollout(policy):
                for key, tran in zip(self.keys, transition):
                    self.paths[key].append(tran)
            self.n_path += 1
        for key in self.keys:
            self.paths[key] = np.asarray(self.paths[key])
            if self.paths[key].ndim == 1:
                self.paths[key] = np.expand_dims(self.paths[key], axis=-1)
        self.paths['n_path'] = self.n_path

    def get_adv(self, policy, val_fn, discount, lam):
        values = val_fn.predict(self.paths['states'])
        advantages = np.empty_like(self.paths['reward'])
        for k, val in enumerate(reversed(self.paths['reward'])):
            t = len(self.paths['reward']) - k - 1
            if self.paths['done']: #
                advantages[t] = self.paths['reward'][k] - values[k]
            else:
                sigma = self.paths['reward'][k] + discount * values[k+1] - values[k]
                advantages[t] = sigma + lam * advantages[k+1]
        return advantages, values
