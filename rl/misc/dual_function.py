import numpy as np
from rl.misc.utilies import stable_log_exp_sum
from rl.misc.utilies import discount_norm_rewards
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import minimize
from functools import partial
from collections import defaultdict
#
def dual_function_gradient(epsilon, param_eta, param_v, transitions):
    #
    N = len(transitions.reward)
    rewards = np.array(transitions.reward).reshape((N,))
    rewards = discount_norm_rewards(rewards, gamma=0.95)
    # rewards = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards)) * 1.
    features = np.array(transitions.features).reshape((N, -1))
    next_features = np.array(transitions.next_features).reshape((N, -1))
    features_diff = next_features - features
    #
    param_v = param_v.reshape(len(param_v))
    x0 = np.hstack([param_eta, param_v])
    bounds = [(-np.inf, np.inf) for _ in x0]
    bounds[0] = (1e-5, np.inf)
    #
    def dual_fn(rewards, features_diff, inputs):
        param_eta = inputs[0]
        param_v = inputs[1:]
        advantages = rewards + np.dot(features_diff, param_v)
        weights = advantages / param_eta
        Z = np.exp(advantages / param_eta)
        g = param_eta * epsilon + param_eta * (stable_log_exp_sum(x=weights) + np.log(1. / len(rewards)))
        grad_eta = epsilon + stable_log_exp_sum(x=weights) + np.log(1./len(rewards)) - Z.dot(advantages) / (param_eta * np.sum(Z))
        grad_v = Z.dot(features_diff) / np.sum(Z)
        return g, np.hstack([grad_eta, grad_v])
    #
    opt_fn = partial(dual_fn, rewards, features_diff)
    results = minimize(opt_fn, x0, method="L-BFGS-B", jac=True, options={'disp':False}, bounds=bounds)
    param_eta = results.x[0]
    param_v = results.x[1:]
    X = (rewards + np.dot(features_diff, param_v)) / param_eta
    max_X = np.max(X)
    weights = np.exp(X - max_X) / np.sum(np.exp(X-max_X))
    return param_eta, param_v, weights













































