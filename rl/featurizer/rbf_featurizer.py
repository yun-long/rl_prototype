import numpy as np
import matplotlib.pyplot as plt

class RBFFeaturizer(object):
    def __init__(self, env, dim_features=10, beta=20):
        self.beta = beta
        self.obs_dims = env.observation_space.shape[0]
        self.num_features = dim_features * self.obs_dims
        self.dim_features = dim_features
        self.obs_low = env.observation_space.low
        self.obs_high = env.observation_space.high
        self.norm_low = np.ones(self.obs_dims) * -1
        self.norm_high = np.ones(self.obs_dims) * 1

    def normalizer(self, state):
        norm_state = (state - self.obs_low) / (self.obs_high - self.obs_low)
        norm_state = norm_state * 2 - 1
        return norm_state

    def transform(self, state):
        norm_state = self.normalizer(state)
        centers = np.array([i * (self.norm_high-self.norm_low) / (self.dim_features-1) + self.norm_low for i in range(self.dim_features)])
        phi = np.exp(-self.beta*(norm_state - centers) ** 2).reshape(self.num_features)
        # print(phi.shape)
        return phi

    def plot_1dim(self, ax, x, y):
        for i_features in range(self.dim_features):
            ax.plot(x, y[:, i_features])
            ax.set_xlabel("states")
            ax.set_ylabel("RBF features")

    def plot_Ndim(self,axes,  x, y):
        for i_dim, ax in enumerate(axes.flatten()):
            self.plot_1dim(ax=ax, x=x[:,i_dim], y=y[:, :, i_dim])

    def plot_examples(self, show=True):
        N = 1000
        y_features = []
        states = np.zeros(shape=(N, self.obs_dims))
        for i_dim in range(self.obs_dims):
            states[ :, i_dim ] = np.linspace(self.obs_low[i_dim], self.obs_high[i_dim], N)
        for i_state, state in enumerate(states):
            features = self.transform(state)
            y_features.append(features)
        y_features = np.array(y_features)
        if self.obs_dims == 1:
            fig, ax = plt.subplots(1,1)
            self.plot_1dim(ax=ax, x=states, y=y_features)
        else:
            fig, axes = plt.subplots(nrows=self.obs_dims, ncols=1)
            self.plot_Ndim(axes=axes, x=states, y=y_features)
        if show == True:
            plt.show()

