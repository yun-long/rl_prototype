import numpy as np
from rl.misc.utilies import stable_log_exp_sum
from rl.misc.utilies import discount_norm_rewards
from scipy.optimize import fmin_l_bfgs_b
from functools import partial

def dual_function_gradient(epsilon, param_eta, param_v, transitions):
    #
    N = len(transitions.reward)
    rewards = np.array(transitions.reward).reshape((N,))
    # rewards = discount_norm_rewards(rewards=rewards, gamma=1.0)
    # rewards = np.exp((rewards-max(rewards)) * 2 / (max(rewards) - min(rewards)))
    # rewards -= np.mean(rewards)
    features = np.array(transitions.features).reshape((-1, N))
    next_features = np.array(transitions.next_features).reshape((-1, N))
    features_diff = next_features - features
    #
    param_v = param_v.reshape(len(param_v))
    x0 = np.hstack([param_eta, param_v])
    bounds = [(-np.inf, np.inf) for _ in x0]
    bounds[0] = (0., np.inf)
    #
    def dual_fn(rewards, features_diff, inputs):
        param_eta = inputs[0]
        param_v = inputs[1:]
        td_error = rewards + np.dot(features_diff.T, param_v)
        weights = td_error / param_eta
        g = param_eta * epsilon + param_eta * stable_log_exp_sum(x=weights, N=len(rewards))
        return g
    #
    def dual_grad(rewards, features_diff, inputs):
        param_eta = inputs[0]
        param_v = inputs[1:]
        td_error = rewards + np.dot(features_diff.T, param_v)
        Z = np.exp(td_error / param_eta)
        grad_eta = epsilon + np.log(np.sum(Z) / len(Z)) - Z.dot(td_error) / (param_eta * np.sum(Z))
        grad_theta = Z.dot(features_diff.T) / np.sum(Z)
        return np.hstack([grad_eta, grad_theta])
    #
    opt_fn = partial(dual_fn, rewards, features_diff)
    grad_opt_fn = partial(dual_grad, rewards, features_diff)
    params_new, _, _ = fmin_l_bfgs_b(func=opt_fn,
                                     x0=x0,
                                     bounds=bounds,
                                     fprime=grad_opt_fn,
                                     maxiter=100,
                                     disp=False)

    param_eta = params_new[0]
    param_v = params_new[1:]
    td_error = rewards + np.dot(features_diff.T, param_v)
    weights = np.exp(td_error / param_eta)
    return param_eta, param_v, weights


