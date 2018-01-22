import numpy as np

class GPLinearMean(object):

    def __init__(self, env, featurizer):
        #
        self.num_features = featurizer.num_features
        self.num_actions = env.action_space.shape[0]
        self.featurizer = featurizer
        #
        self.Theta = np.random.randn(self.num_features, self.num_actions) / np.sqrt(self.num_features)
        self.Sigma_theta = np.eye(self.num_features * self.num_actions) * 1e-2 # for exploration in parameter space
        # Gaussian noise, for exploration in action space
        self.Mu_noise = np.zeros(self.num_actions) * 0
        self.Sigma_noise = np.eye(self.num_actions) * 1e-2

    def sample_theta(self, num_samples):
        """
        Exploration in paramter space, used for Episode-based usually
        :param num_samples:
        :return:
        """
        Mu = self.Theta.reshape(self.num_actions*self.num_features)
        theta_samples = np.random.multivariate_normal(mean=Mu,
                                                      cov=self.Sigma_theta,
                                                      size=num_samples)
        return theta_samples

    def predict_action(self, state, theta):
        """
        Predict action according to the parameter samples. Episode-based usually
        :param state:
        :param theta:
        :return:
        """
        theta = np.reshape(theta, self.Theta.shape)
        featurized_state = self.featurizer.transform(state)
        action = np.dot(theta.T, featurized_state)
        return action


    def sample_action(self, state):
        """
        Exploration in action_space, used for Step-based usually.
        :param state:
        :return:
        """
        featurized_state = self.featurizer.transform(state)
        noise = np.random.multivariate_normal(mean=self.Mu_noise,
                                              cov=self.Sigma_noise,
                                              size=1)
        action = np.dot(self.Theta.T, featurized_state) + noise
        return action, noise

    def update_pg(self):
        pass

    def update_em(self, Weights, Phi, A):
        """

        :param Weights: Weighted rewards,
        :param Phi: Features of sampled states
        :param A: Actions
        :return: None
        """
        H, T, N = Phi.shape
        Weights = Weights.reshape(T*H)
        Weights = np.diag(Weights)
        A = A.reshape(T*H)
        theta = np.zeros_like(self.Theta)
        for n in range(N):
            phi_n = Phi[:,:,n]
            phi_n = phi_n.reshape(T*H)
            theta_tmp = np.dot(phi_n.T, Weights)
            theta_tmp = np.dot(theta_tmp, phi_n)
            theta_tmp = 1./ theta_tmp
            # theta_tmp = np.atleast_2d(theta_tmp)
            # theta_tmp = np.linalg.inv(theta_tmp)
            theta_tmp = np.dot(theta_tmp, phi_n.T)
            theta_tmp = np.dot(theta_tmp, Weights)
            theta_tmp = np.dot(theta_tmp, A)
            theta[n, :] = theta_tmp
        self.Theta = theta


    def update_em2(self, Weights, Phi, A):
        H = Weights.shape[0]
        T = Weights.shape[1]
        phi = Phi.reshape(( H*T, self.num_features))
        Q = Weights.reshape((H*T))
        Q = np.diag(Q)
        A = A.reshape((H*T, 1))
        theta_tmp = np.dot(phi.T, Q)
        theta_tmp = np.dot(theta_tmp, phi)
        theta_tmp = np.linalg.inv(theta_tmp)
        theta_tmp = np.dot(theta_tmp, phi.T)
        theta_tmp = np.dot(theta_tmp, Q)
        theta_tmp = np.dot(theta_tmp, A)
        self.Theta = theta_tmp
