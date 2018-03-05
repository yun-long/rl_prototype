import numpy as np

class AdvancedSampler(object):

    def __init__(self, env, sess=None):
        self.env = env
        self.keys = ['state', 'action', 'reward', 'done']
        self.sess = sess

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
            if self.sess is not None:
                action = policy(state, self.sess)
            else:
                action = policy(state)
            # print(action)
            next_state, reward, done, _ = self.env.step(action)
            yield state, action, reward, done
            state = next_state

    def rollous(self, policy, n_trans):
        paths, n_path = self.create_path()
        while len(paths['reward']) < n_trans:
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

    def get_adv(self, paths, val_fn, discount, lam):
        values = val_fn(paths['state'])
        advantages = np.empty_like(paths['reward'])
        for rev_t, val in enumerate(reversed(paths['reward'])):
            t = len(values) - rev_t - 1
            if paths['done'][t]:
                advantages[t] = paths['reward'][t] - values[t]
            else:
                sigma = paths['reward'][t] + discount * values[t+1] - values[t]
                advantages[t] = sigma + lam * advantages[t+1]
        return advantages, values

def next_batch_idx(batch_size, data_set_size):
    batch_idx_list = np.random.choice(data_set_size, data_set_size, replace=False)
    for batch_start in range(0, data_set_size, batch_size):
        yield batch_idx_list[batch_start: min(batch_start+batch_size, data_set_size)]
