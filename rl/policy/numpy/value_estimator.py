import numpy as np
import matplotlib.pyplot as plt

class ValueEstimator(object):

    def __init__(self, featurizer):
        self.num_features = featurizer.num_features
        self.num_output = 1
        self.param_v = np.random.rand(self.num_features, self.num_output) / np.sqrt(self.num_features)
        self.featurizer = featurizer

    def predict(self, state):
        featurized_state = self.featurizer.transform(state)
        value = np.dot(self.param_v.T, featurized_state)
        return value

    def update(self, new_param_v):
        self.param_v = new_param_v
        return True

    def plot_1D(self, env, show=True):
        fig = plt.figure()
        y_value = []
        x_state = []
        for state in np.arange(env.observation_space.n):
            x_state.append(state)
            y_value.append(self.predict(state))
        plt.plot(x_state, y_value)
        plt.xlabel("States")
        plt.ylabel("Values")
        plt.title("Value function")
        if show:
            plt.show()
        return fig