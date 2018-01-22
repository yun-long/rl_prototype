
class GaussianPolicy(object):

    def __init__(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def sample_theta(self, num_samples):
        raise NotImplementedError
