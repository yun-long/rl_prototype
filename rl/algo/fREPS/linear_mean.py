"""
An implementation of f-REPS on random jumpy environment, Gaussian policy, linear mean.
"""
from rl.policy.gp_linear_mean import GPLinearMean
from rl.featurizer.non_featurizer import NoneFeaturizer
from rl.featurizer.poly_featurizer import PolyFeaturizer
from rl.policy.value_estimator import ValueEstimator
from rl.misc.dual_function import optimize_fdual_fn_v2
from rl.sampler.standard_sampler import StandardSampler
from rl.env.random_jump import RandomJumpEnv
from rl.misc.plot_rewards import plot_tr_ep_rs
#
import numpy as np
#
env = RandomJumpEnv()
#
print("Action space : ", env.action_space)
print("Action space low : ", env.action_space.low)
print("Action space high : ", env.action_space.high)
print("Observation space : ", env.observation_space)
print("Observation space low : ", env.observation_space.low)
print("Observation space high : ", env.observation_space.high)
#
pol_featurizer = NoneFeaturizer(env)
#
val_featurizer = PolyFeaturizer(env, degree=2)
#
sampler = StandardSampler(env)
#
num_episodes = 50
num_trials = 2
num_samples = 500
#

for i_trail in range(num_trials):
    #
    policy = GPLinearMean(env, pol_featurizer)
    value = ValueEstimator(val_featurizer)
    #
    v0 = np.squeeze(value.param_v0)
    eta0 = 5.0
    #
    for i_episode in range(num_episodes):
        #
        sample_data, mean_rs = sampler.sample_data(policy=policy, N=num_samples)
        #
        rewards, val_feat_diff, A, Phi = sampler.process_data(sample_data, pol_featurizer, val_featurizer)
        #
        _, _, weights = optimize_fdual_fn_v2(rewards, val_feat_diff, eta0, v0)