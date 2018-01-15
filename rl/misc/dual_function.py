import numpy as np
from scipy.optimize import fmin_l_bfgs_b


class DualFunction(object):

    def __init__(self, eta_init, v_init, epsilon):
        self.x0 = np.hstack([eta_init, v_init])
        self.bounds = [(-np.inf, np.inf) for _ in self.x0]
        self.bounds[0] = (0., np.inf)
        self.epsilon = epsilon

    def update(self, rewards, features, next_features):
        N = len(rewards)
        self.rewards = rewards.reshape((N,))
        self.features = np.array(features).reshape((-1, N))
        self.next_features = np.array(next_features).reshape((-1, N))
        self.features_diff = self.next_features - self.features
        eta, v = self.dual_optimize()
        return eta, v

    def stable_log_exp_sum(self, x, N=None):
        """
        y = np.log(np.sum(np.exp(x)) / len(x)) # not stable
          = np.max(x) + np.log(np.sum(np.exp(x - np.max(x))) / len(x)) # stable
        :param x:
        :return:
        """
        max_x = np.max(x)
        if N is None:
            y = max_x + np.log(np.sum(np.exp(x-max_x)))
        else:
            y = max_x + np.log(np.sum(np.exp(x-max_x)) / N)
        return y

    def td_target(self, v):
        return self.rewards + np.dot(v, self.features_diff)

    def Z(self, v, eta):
        return self.td_target(v) / eta

    def dual_fn(self, inputs):
        param_eta = inputs[0]
        param_v = inputs[1:]
        weights = self.td_target(param_v) / param_eta
        g = param_eta * self.epsilon + param_eta * self.stable_log_exp_sum(x=weights)
        return g

    def dual_grad(self, inputs):
        param_eta = inputs[0]
        param_v = inputs[1:]
        Z = self.Z(param_v, param_eta)
        grad_eta = self.epsilon + np.log(np.sum(Z) / len(Z)) - Z.dot(self.td_target(param_v)) / (param_eta * np.sum(Z))
        grad_theta = Z.dot(self.features_diff.T) / np.sum(Z)
        return np.hstack([grad_eta, grad_theta])

    def dual_optimize(self):
        params_new, _, _ = fmin_l_bfgs_b(func=self.dual_fn,
                                         x0=self.x0,
                                         bounds=self.bounds,
                                         fprime=self.dual_grad,
                                         maxiter=100,
                                         disp=False)

        self.eta = params_new[0],
        self.v = params_new[1:]
        self.x0 = np.hstack([self.eta, self.v])
        return self.eta, self.v
