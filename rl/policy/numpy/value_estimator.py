import numpy as np


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
