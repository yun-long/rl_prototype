import numpy as np


class ValueEstimatorNP(object):

    def __init__(self, featurizer, learning_rate = 0.001):
        self.num_featuries = featurizer.num_featuries
        self.num_output = 1
        self.v = np.random.rand(self.num_featuries, self.num_output) / np.sqrt(self.num_featuries)
        self.featurizer = featurizer
        self.learning_rate = learning_rate

    def predict(self, state):
        featurized_state = self.featurizer.transform(state)
        value = np.dot(self.v.T, featurized_state)
        return value

    def update_reps(self, new_theta):
        self.v = new_theta
        return True
