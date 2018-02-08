import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

class PolyFeaturizer(object):

    def __init__(self, env, degree=2):
        self.env = env
        self.degree = degree
        # number of features
        self.num_features = self.fit()
        self.obs_dims = env.observation_space.shape[0]
        self.obs_low = env.observation_space.low
        self.obs_high = env.observation_space.high
        self.norm_low = np.ones(self.obs_dims) * -10
        self.norm_high = np.ones(self.obs_dims) * 10

    def fit(self):
        init_obs = self.env.reset()
        init_obs = np.atleast_2d(init_obs)
        dim_features = PolynomialFeatures(degree=self.degree).fit_transform(init_obs).shape[1]
        return dim_features

    def normalizer(self, state):
        norm_state = (state - self.obs_low) / (self.obs_high - self.obs_low)
        norm_state = norm_state * 2 - 1
        return norm_state

    def transform(self, state):
        state = np.reshape(state, (-1, self.obs_dims))
        state = self.normalizer(state)
        phi = PolynomialFeatures(degree=self.degree).fit_transform(state).reshape((self.num_features,))
        return phi

    def plot_1dim(self, ax, x, y):
        for i_features in range(self.num_features):
            ax.plot(x, y[:, i_features])
            ax.set_xlabel("states")
            ax.set_ylabel("Polynomial features")

    def plot_Ndim(self, axes, x, y):
        for i_dim, ax in enumerate(axes.flatten()):
            self.plot_1dim(ax=ax, x=x[:, i_dim], y=y[:, :, i_dim])

    def plot_examples(self, show=True):
        N = 1000
        y_features = []
        states = np.zeros(shape=(N, self.obs_dims))
        for i_dim in range(self.obs_dims):
            states[:, i_dim] = np.linspace(self.obs_low[i_dim], self.obs_high[i_dim], N)
        for i_state, state in enumerate(states):
            features = self.transform(state)
            features = np.reshape(features, newshape=(self.num_features, self.obs_dims))
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
