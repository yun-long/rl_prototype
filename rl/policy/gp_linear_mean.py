"""
Gaussin policy, linear mean, constant variance

Reference: Jan Peters, A Survey on policy search for robotics
"""

import numpy as np
import time
from rl.policy.base import GaussianPolicy

class GPLinearMean(GaussianPolicy):

    def __init__(self, env, featurizer):
        #
        self.env = env
        #
        self.num_features = featurizer.num_features
        self.num_actions = env.action_space.shape[0]
        self.featurizer = featurizer
        #
        self.Mu_theta = np.random.randn(self.num_features, self.num_actions) / np.sqrt(self.num_features)
        self.Sigma_action = np.eye(self.num_actions) * 1e1 # for exploration in parameter space
        super().__init__()

    def predict_action(self, state):
        """
        Exploration in action_space, used for Step-based usually.
        :param state:
        :return:
        """
        featurized_state = self.featurizer.transform(state).T
        Mu_action = np.dot(self.Mu_theta.T, featurized_state).reshape(self.num_actions)
        try:
            action = np.random.multivariate_normal(Mu_action, self.Sigma_action)
        except:
            raise ValueError
        return action

    def update_pg(self, alpha_coeff, theta_samples, advantanges):
        pass


    def update_wml(self, Weights, Phi, A):
        T = Phi.shape[0]
        phi = Phi.reshape((T, self.num_features))
        Q = Weights.reshape(T)
        Q = np.diag(Q)
        A = A.reshape((T, self.num_actions))
        theta_tmp1 = np.linalg.inv(np.dot(phi.T, np.dot(Q, phi)))
        theta_tmp2 = np.dot(phi.T, np.dot(Q, A))
        self.Mu_theta = np.dot(theta_tmp1, theta_tmp2).reshape(self.Mu_theta.shape)
        #
        Z = (np.sum(Weights)**2 - np.sum(Weights**2)) / np.sum(Weights)
        nume_sum = 0
        for i in range(len(Weights)):
            tmp = np.outer((A[i] - np.dot(self.Mu_theta.T, phi[i, :])), (A[i] - np.dot(self.Mu_theta.T, phi[i, :])))
            tmp = Weights[i] * tmp
            nume_sum += tmp
        self.Sigma_action = nume_sum / Z

    def optimal_policy_demo(self, num_demos):
        for i_demo in range(num_demos):
            print("Optimal Policy Demo : ", i_demo)
            state = self.env.reset()
            while True:
                action = self.predict_action(state)
                next_state, rewards, done, _ = self.env.step(action)
                state = next_state
                self.env.render()
                if done:
                    time.sleep(1)
                    break
        self.env.render(close=True)

