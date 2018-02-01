import numpy as np

class GPLinearMean(object):

    def __init__(self, env, featurizer):
        #
        self.num_features = featurizer.num_features
        self.num_actions = env.action_space.shape[0]
        self.featurizer = featurizer
        #
        self.Mu_theta = np.random.randn(self.num_features, self.num_actions) / np.sqrt(self.num_features)
        self.Sigma_action = np.eye(self.num_actions) * 1e4 # for exploration in parameter space

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

    def update_pg(self):
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
