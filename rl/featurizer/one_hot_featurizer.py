import numpy as np
import matplotlib.pyplot as plt

class OneHotFeaturizer(object):

    def __init__(self, env):
        self.env = env
        self.num_features = env.observation_space.n
        self.obs_dims = env.observation_space.n


    def transform(self, state):
        state_one_hot = np.zeros(self.num_features)
        state_one_hot[state] = 1
        return state_one_hot

    def print_examples(self, show=True):
        N = self.num_features
        y_features = []
        x_states = []
        print("\nOneHotFeaturizer example of discrete environment.")
        for state in range(N):
            state_features = self.transform(state)
            x_states.append(state)
            y_features.append(state_features)
            print("state : ", state, ", features : ", state_features)


