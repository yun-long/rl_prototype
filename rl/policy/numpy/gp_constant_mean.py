import numpy as np
from rl.policy.numpy.base import GaussianPolicy

class GPConstantMean(object):

    def __init__(self, num_dim):
        self.num_dim = num_dim
        self.Mu = np.zeros(shape=num_dim)
        self.Sigma = np.eye(num_dim) * 1e3

    def sample_theta(self, num_samples):
        theta_samples = np.random.multivariate_normal(mean=self.Mu,
                                                      cov=self.Sigma,
                                                      size=num_samples)
        return theta_samples

    def update_pg(self, alpha_coeff, theta_samples, advantages):
        Std_w = np.diag(self.Sigma)
        diff = theta_samples - self.Mu
        d_log_pi_Mu = diff * (1. / Std_w*2)
        d_log_pi_Std = diff * 2 / Std_w**3 - 1./Std_w
        d_log_pi_Omega = np.hstack((d_log_pi_Mu, d_log_pi_Std))
        G = np.dot(advantages, d_log_pi_Omega) / len(theta_samples)
        # Normalize variance gradient
        G_sigma = G[self.num_dim:]
        G_sigma = G_sigma / np.linalg.norm(G_sigma)
        G[self.num_dim:] = G_sigma
        #
        self.Mu = self.Mu + alpha_coeff * G[:self.num_dim]
        self.Sigma = self.Sigma + alpha_coeff * np.diag(G[self.num_dim:])

    def update_em(self, theta_samples, weights):
        self.Mu = weights.dot(theta_samples)/np.sum(weights)
        Z = (np.sum(weights)**2-np.sum(weights**2))/np.sum(weights)
        self.Sigma = np.sum([weights[i]*(np.outer((theta_samples[i]-self.Mu), (theta_samples[i]-self.Mu))) for i in range(len(weights))], 0)/Z
