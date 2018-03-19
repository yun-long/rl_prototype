import numpy as np
#
from gym.spaces.tuple_space import Tuple
from gym.spaces.discrete import Discrete
from rl.misc.memory import Transition
from collections import defaultdict

class StandardSampler(object):

    def __init__(self, env):
        if isinstance(env.action_space, Tuple):
            self.act_type = 'tuple'
        elif isinstance(env.action_space, Discrete):
            self.act_type = 'discrete'
        else:
            raise NotImplementedError
        self.env = env

    def sample_data(self, policy, N):
        """
        Collect N samples (s, a, s', r) by following policy pi
        :param policy: np array (num_states, num_actions)
        :param N: number of sample steps
        :return: a collection of namedtuple (state, action, next_state, reward)
        """
        data = []
        state = self.env.reset()
        mean_reward = []
        for i in range(N):
        # while True:
            if self.act_type == 'tuple':
                action_idx, action  = policy.predict_action(state)
                next_state, reward, done, _ = self.env.step(action)
                data.append(Transition(state=state,
                                       action=action_idx,
                                       next_state=next_state,
                                       reward=reward))
            elif self.act_type == 'discrete':
                action = policy.predict_action(state)
                next_state, reward, done, _ = self.env.step(action)
                data.append(Transition(state=state,
                                       action=action,
                                       next_state=next_state,
                                       reward=reward))
            else:
                raise NotImplementedError
            mean_reward.append(reward)
            if done:
                next_state = self.env.reset()
                # break
            state = next_state
        return data, np.mean(mean_reward)

    def count_data(self, data, featurizer):
        """
        For REPS discrete case
        :param data:
        :param featurizer:
        :return:
        """
        n, r, features_diff = defaultdict(float), defaultdict(float), defaultdict(float)
        for t_i, transition in enumerate(data):
            state_action = (transition.state, transition.action)
            r[state_action] += transition.reward
            n[state_action] += 1
            features_diff[state_action] += featurizer.transform(transition.next_state) \
                                           - featurizer.transform(transition.state)
        sa_pairs, r_array, features_diff_array, sa_pairs_n = [], [], [], []
        for key in sorted(n.keys()):
            sa_pairs_n.append(n[key])
            sa_pairs.append(key)
            r_array.append(r[key] / n[key])
            features_diff_array.append(features_diff[key] / n[key])
        return np.array(r_array), np.array(features_diff_array), np.array(sa_pairs_n), sa_pairs

    def process_data(self, data, pol_featurizer, val_featurizer):
        """
        For REPS continuous case
        :param data:
        :param pol_featurizer:
        :param val_featurizer:
        :return:
        """
        N = len(data)
        rewards, val_feat_diff = [],[]
        actions, pol_feat = [], []
        for t_i, transition in enumerate(data):
            actions.append(transition.action)
            pol_feat.append(pol_featurizer.transform(transition.state))
            #
            rewards.append(transition.reward)
            val_feat_diff.append(val_featurizer.transform(transition.next_state) -
                                 val_featurizer.transform(transition.state))
        val_feat_diff = np.array(val_feat_diff).reshape((N, -1))
        rewards = np.array(rewards).reshape((N,))
        actions = np.array(actions)
        pol_feat = np.array(pol_feat)
        return rewards, val_feat_diff, actions, pol_feat
