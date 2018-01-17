import numpy as np

class GaussianPolicyNP(object):

    def __init__(self, env, featurizer):
        self.num_features = featurizer.num_features
        self.num_actions = env.action_space.shape[0]
        # xavier initialization
        self.theta = np.random.randn(self.num_features, self.num_actions) / np.sqrt(self.num_actions)
        self.theta_sigma = np.eye(self.num_features) * 1e-1
        # for exploration
        self.action_sigma = np.eye(self.num_actions) * 1e-2
        self.featurizer = featurizer

    def samples_theta(self, num_samples):
        theta_samples = np.random.multivariate_normal(self.theta[:,0], self.theta_sigma, num_samples)
        return theta_samples

    def update_episode(self, weights, theta_samples):
        weights = np.array(weights)
        weights = np.mean(weights, axis=1)
        self.theta = weights.T.dot(theta_samples) / np.sum(weights)
        self.theta = self.theta[:, None]

    def update_step(self, weights, Phi_features, actions):
        Phi_features = np.array(Phi_features)
        actions = np.array(np.squeeze(actions))
        weights = np.array(weights).reshape(len(actions))
        Weights = np.diag(weights)
        theta_new = np.dot(np.dot(Phi_features.T, Weights), Phi_features)
        # print(theta_new)
        theta_new = np.linalg.pinv(theta_new)
        theta_new = np.dot(theta_new, Phi_features.T)
        theta_new = np.dot(theta_new, Weights)
        self.theta = np.dot(theta_new, actions).reshape(self.theta.shape)

    def predict_step(self, state):
        featurized_state = self.featurizer.transform(state)
        action_mu = np.dot(self.theta.T, featurized_state)
        action = np.random.multivariate_normal(action_mu, self.action_sigma, 1)
        return action

    def predict_episode(self, state, theta_sample):
        featurized_state = self.featurizer.transform(state)
        action = np.dot(theta_sample[:,None].T, featurized_state)
        return action

