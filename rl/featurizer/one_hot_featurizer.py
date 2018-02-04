import numpy as np
import matplotlib.pyplot as plt

class OneHotFeaturizer(object):

    def __init__(self, env):
        self.env = env
        self.num_features = env.observation_space.n
        self.obs_dims = env.observation_space.n
        self.phi = np.eye(self.num_features)

    def transform(self, state):
        state_one_hot = self.phi[state]
        return state_one_hot

    def print_examples(self):
        print("\nOneHotFeaturizer example of discrete environment.")
        for state in range(self.num_features):
            state_features = self.transform(state)
            print("state : ", state, ", features : ", state_features)


