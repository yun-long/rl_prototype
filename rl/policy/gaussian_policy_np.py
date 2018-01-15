import numpy as np

class GaussianPolicyNP(object):

    def __init__(self, env, featurizer):
        self.num_featuries = featurizer.num_featuries
        self.num_actions = env.action_space.shape[0]
        # xavier initialization
        self.theta = np.random.randn(self.num_featuries, self.num_actions) / np.sqrt(self.num_actions)
        self.theta_sigma = np.eye(self.num_featuries) * 1e-1
        #
        # self.Mu = np.zeros(self.num_actions)
        self.Sigma = np.eye(self.num_actions) * 1e-1
        self.featurizer = featurizer

    def samples(self, num_samples):
        # theta_samples = np.zeros((num_samples, self.num_featuries))
        theta_samples = np.random.multivariate_normal(self.theta[:,0], self.theta_sigma, num_samples)
        return theta_samples

    def update(self, weights, theta_samples):
        weights = np.array(weights)
        weights = np.mean(weights, axis=1)
        self.theta = weights.T.dot(theta_samples) / np.sum(weights)
        self.theta = self.theta[:, None]

    def predict(self, state, theta_sample):
        featurized_state = self.featurizer.transform(state)
        Mu_state = np.dot(theta_sample[:, None].T, featurized_state)
        action = np.random.multivariate_normal(Mu_state, self.Sigma)
        return action