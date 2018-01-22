import numpy as np

class GaussianPolicyNP(object):

    def __init__(self, env, featurizer):
        self.num_features = featurizer.num_features
        self.num_actions = env.action_space.shape[0]
        # xavier initialization
        self.theta = np.random.randn(self.num_features, self.num_actions) / np.sqrt(self.num_actions)
        self.theta_sigma = np.eye(self.num_features) * 1e-1
        # for exploration
        self.action_sigma = np.eye(self.num_actions) * 1e-1
        self.featurizer = featurizer

    def samples_theta(self, num_samples):
        theta_samples = np.random.multivariate_normal(self.theta[:,0], self.theta_sigma, num_samples)
        return theta_samples

    def samples_actions(self, state, num_samples):
        featurized_state = self.featurizer.transform(state)
        # self.theta = self.theta.reshape(self.theta.shape)
        mu_action = np.dot(self.theta.T, featurized_state)
        mu_action = np.atleast_1d(mu_action)
        action_samples = np.random.multivariate_normal(mu_action, self.action_sigma, num_samples)
        return action_samples

    def update_episode(self, weights, theta_samples):
        # weights = np.array(weights)
        # weights = np.mean(weights, axis=1)
        self.theta = weights.T.dot(theta_samples)
        self.theta = self.theta[:, None]

    # def update_step(self, weights, Phi_features, actions):
    #     # TODO: Update Rule
    #     theta_update = 0
    #     for i in range(weights.shape[1]):
    #         phi_features = Phi_features[:,:,i]
    #         Weights = np.diag(weights[:,i])
    #         U = actions[:,i]
    #         theta_new = np.dot(np.dot(phi_features.T, Weights), phi_features)
    #         theta_new = np.linalg.pinv(theta_new)
    #         theta_new = np.dot(theta_new, phi_features.T)
    #         theta_new = np.dot(theta_new, Weights)
    #         theta_new = np.dot(theta_new, U).reshape(self.theta.shape)
    #         theta_update += theta_new
    #     self.theta = theta_update / weights.shape[1]

    # def update_step(self, weights, Phi, Actions):
    #     T = Phi.shape[1]
    #     sum_1 = 0
    #     sum_2 = 0
    #     for t in range(T):
    #         q_i = weights[:,t,0]
    #         phi = Phi[:,t, :]
    #         a = Actions[:, t, 0]
    #         sum_1 += np.dot(np.dot(q_i, phi), phi.T)
    #         sum_2 += np.dot(np.dot(a, q_i), phi)
    #     self.theta = (np.mean(sum_2, axis=0) / np.mean(sum_1)).reshape(self.theta.shape)

    def update_step(self, weights, Phi, Actions):
        H = weights.shape[1]
        T = weights.shape[0]
        phi = Phi.reshape(( H*T, self.num_features))
        Q = weights.reshape((H*T))
        Q = np.diag(Q)
        A = Actions.reshape((H*T, 1))
        theta_tmp = np.dot(phi.T, Q)
        theta_tmp = np.dot(theta_tmp, phi)
        theta_tmp = np.linalg.inv(theta_tmp)
        theta_tmp = np.dot(theta_tmp, phi.T)
        theta_tmp = np.dot(theta_tmp, Q)
        theta_tmp = np.dot(theta_tmp, A)
        self.theta = theta_tmp

    def predict_step(self, state):
        featurized_state = self.featurizer.transform(state)
        action_mu = np.dot(self.theta.T, featurized_state)
        # print(action_mu)
        action = np.random.multivariate_normal(action_mu, self.action_sigma, 1)
        return action

    def predict_episode(self, state, theta_sample):
        featurized_state = self.featurizer.transform(state)
        action = np.dot(theta_sample[:,None].T, featurized_state)
        return action

