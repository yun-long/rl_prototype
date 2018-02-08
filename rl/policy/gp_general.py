import numpy as np
import matplotlib.pyplot as plt


class GPGeneral(object):

    def __init__(self, env, featurizer):
        self.env = env
        self.n_a = env.action_space.shape[0]
        self.n_s = env.observation_space.shape[0]
        self.featurizer = featurizer
        self.theta_mu = np.ones((self.n_a, self.featurizer.num_features))
        self.theta_sigma = np.ones((self.n_a, self.featurizer.num_features))

    def predict(self, state):
        x_s = self.featurizer.transform(state)
        mu_s_theta = np.dot(self.theta_mu, x_s)
        sigma_s_theta = np.exp(np.dot(self.theta_sigma, x_s))
        action = np.random.multivariate_normal(mean=mu_s_theta, cov=np.eye(sigma_s_theta.shape[0])*sigma_s_theta)
        return action, x_s, mu_s_theta, sigma_s_theta

    def update_mu(self, A, Mu_s_theta, Sigma_s_theta, X_s, R, alpha):
        grad_mu = A - Mu_s_theta
        grad_mu = grad_mu / np.square(Sigma_s_theta)
        grad_mu = grad_mu * X_s
        grad_mu = grad_mu.T * R
        self.theta_mu += alpha * grad_mu

    def update_sigma(self, A, Mu_s_theta, Sigma_s_theta, X_s, R, alpha):
        grad_sigma = np.square(A - Mu_s_theta) / np.square(Sigma_s_theta) - 1
        grad_sigma = grad_sigma *  X_s
        grad_sigma = grad_sigma * R
        self.theta_sigma += alpha * grad_sigma
