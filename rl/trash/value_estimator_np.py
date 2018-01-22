import numpy as np


class ValueEstimatorNP(object):

    def __init__(self, featurizer, learning_rate = 0.001):
        self.num_features = featurizer.num_features
        self.num_output = 1
        self.param_v = np.random.rand(self.num_features, self.num_output) / np.sqrt(self.num_features)
        self.featurizer = featurizer
        self.learning_rate = learning_rate

    def predict(self, state):
        featurized_state = self.featurizer.transform(state)
        value = np.dot(self.param_v.T, featurized_state)
        return value

    def update_reps(self, new_param_v):
        self.param_v = new_param_v
        return True
