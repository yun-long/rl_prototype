"""
An implementation of REPS on random jumpy environment with Gaussian Policy, Constant mean.
"""

from rl.env.random_jump import RandomJumpEnv
from rl.featurizer.rbf_featurizer import RBFFeaturizer
from rl.policy.numpy.gp_constant_mean import GPConstantMean
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy.optimize import minimize
import itertools

def stable_log_sum_exp(x, N=None):
    """
    y = np.log( np.sum(np.exp(x)) / len(x))  # not stable
      = np.max(x) + np.log(np.sum(np.exp(x-np.max(x)) / len(x))) # stable
    """
    a = np.max(x)
    if N is None:
        y = a + np.log(np.sum(np.exp(x-a)))
    else:
        y = a + np.log(np.sum(np.exp(x-a)) / N)
    return y

def dual_function(epsilon, rewards, eta):
    N = len(rewards)
    x = rewards / eta
    weights = np.exp(x)
    g = eta * epsilon + eta * stable_log_sum_exp(x, N)
    dg = epsilon + stable_log_sum_exp(x, N) - weights.T.dot(rewards) / (eta * np.sum(weights))
    return g, dg

def optimize_dual_function(eps, rewards, x0):
    optfunc = partial(dual_function, eps, rewards)
    result = minimize(optfunc, x0, method="L-BFGS-B", jac=True, options={'disp':False}, bounds=[(1e-5, np.inf)])
    return result.x

env = RandomJumpEnv()
print("observation low : ", env.observation_space.low)
print("observation high : ", env.observation_space.high)
#
eta_init = 5
epsilon = 0.5 # KL bound
num_features = 10
num_trials = 10
num_episodes = 200
num_samples = 20
rbf_featuries = RBFFeaturizer(env, num_features)
#
policy = GPConstantMean(num_dim=num_features)
eta_hat = eta_init

for j in range(num_episodes):
    theta_samples = policy.sample_theta(num_samples=num_samples)
    sample_rewards = []
    for i_theta, theta in enumerate(theta_samples):
        rewards = []
        state = env.reset()
        for t in itertools.count():
            features = rbf_featuries.transform(state=state)
            action = np.atleast_2d(np.dot(theta, features))
            next_obs, reward, done, _ = env.step(action)
            obs = next_obs
            rewards.append(reward)
            if done or t >= 1000:
                # print(done)
                break
        sample_rewards.append(np.sum(rewards))
    sample_rewards = np.array(sample_rewards)
    rewards_normalize = (sample_rewards - np.max(sample_rewards)) / (np.max(sample_rewards) - np.min(sample_rewards))
    eta_hat = optimize_dual_function(epsilon, rewards_normalize, eta_hat)
    weights = np.exp(rewards_normalize / eta_hat)
    policy.update_em(theta_samples, weights)
    print("episode {}, reward {}".format(j, np.mean(sample_rewards)))
    #
    if j >= (num_episodes-1):
        state = env.reset()
        theta = policy.sample_theta(num_samples=1)
        for t in itertools.count():
            env.render()
            features = rbf_featuries.transform(state=state)
            action = np.atleast_2d(np.dot(theta, features))
            next_state, reward, done, _ = env.step(action)
            state = next_state
