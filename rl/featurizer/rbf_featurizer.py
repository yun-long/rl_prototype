import numpy as np
import matplotlib.pyplot as plt

class RBFFeaturizer(object):
    def __init__(self, env, num_features=10, beta=20):
        self.beta = beta
        self.obs_low = env.observation_space.low
        self.obs_high = env.observation_space.high
        self.norm_low = -1
        self.norm_high = 1
        self.num_features = num_features

    def normalizer(self, state):
        norm_state = (state - self.obs_low) / (self.obs_high - self.obs_low)
        norm_state = norm_state * 2 - 1
        return norm_state

    def transform(self, state):
        norm_state = self.normalizer(state)
        centers = np.array([i * (self.norm_high-self.norm_low) / (self.num_features-1) + self.norm_low for i in range(self.num_features)])
        phi = np.exp(-self.beta*(norm_state - centers) ** 2)
        return phi

    def plot_examples(self, show=True):
        N = 1000
        y_features = []
        x_features = []
        for state in np.linspace(self.obs_low, self.obs_high, N):
            x_features.append(state)
            features = self.transform(state)
            y_features.append(features)
        y_features = np.array(y_features)
        fig = plt.figure()
        for i in range(self.num_features):
            plt.plot(x_features, y_features[:, i])
            plt.hold(True)
        if show == True:
            plt.show()

