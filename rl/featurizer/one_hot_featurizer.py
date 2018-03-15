import numpy as np
from gym.spaces.discrete import Discrete
from gym.spaces.tuple_space import Tuple

class OneHotFeaturizer(object):

    def __init__(self, env):
        self.env = env
        if type(env.observation_space) is Discrete:
            self.num_features = env.observation_space.n
            self.obs_dims = env.observation_space.n
        else:
            raise NotImplementedError
        self.phi = np.eye(self.num_features)

    def transform(self, state):
        state_one_hot = self.phi[state]
        return state_one_hot

    def print_examples(self):
        print("\nOneHotFeaturizer example of discrete environment.")
        for state in range(self.num_features):
            state_features = self.transform(state)
            print("state : ", state, ", features : ", state_features)


