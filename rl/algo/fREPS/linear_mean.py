"""
An implementation of f-REPS on random jumpy environment, Gaussian policy, linear mean.
"""
from rl.policy.gp_linear_mean import GPLinearMean
from rl.featurizer.non_featurizer import NoneFeaturizer
from rl.featurizer.poly_featurizer import PolyFeaturizer
from rl.policy.value_estimator import ValueEstimator
from rl.misc.dual_function import optimize_fdual_fn_v2, optimize_dual_fn
from rl.sampler.standard_sampler import StandardSampler
from rl.env.random_jump import RandomJumpEnv
from rl.misc.plot_rewards import plot_tr_ep_rs, plot_coeff_tr_ep_rs
from rl.misc.plot_value import plot_2D_value
#
import numpy as np
from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
#
env = RandomJumpEnv()
# env = Continuous_MountainCarEnv()
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
num_trials = 10
num_samples = 1000
#
alphas = [-10., -5., -2., 0., 1.]
#
mean_rewards = np.zeros(shape=(num_episodes, num_trials, len(alphas)))
for i_alpha, alpha in enumerate(alphas):
    for i_trail in range(num_trials):
        #
        policy = GPLinearMean(env, pol_featurizer)
        value = ValueEstimator(val_featurizer)
        #
        eta0 = 5.0
        epsilon = 0.5
        #
        rnd = np.random.RandomState(seed=43225801)
        env.seed(seed=147691)
        for i_episode in range(num_episodes):
            #
            sample_data, mean_rs = sampler.sample_data(policy=policy, N=num_samples)
            #
            rewards, val_feat_diff, Actions, Phi = sampler.process_data(sample_data, pol_featurizer, val_featurizer)
            #
            if alpha == 1.:
                opt_eta, opt_v, weights = optimize_dual_fn(rewards, val_feat_diff, eta0, epsilon, rnd)
            else:
                opt_lamda, opt_v, weights = optimize_fdual_fn_v2(rewards, val_feat_diff, eta0, alphas[i_alpha], rnd)
            # update policy
            policy.update_wml(weights, Phi, Actions)
            #
            value.update(opt_v)
            #
            print("alpha : {}, Trails : {}, Episode : {}, Reward : {}".format(alpha, i_trail, i_episode, mean_rs))
            mean_rewards[i_episode, i_trail, i_alpha] = mean_rs
        #

plot_2D_value(env, value, conti=True)
#
plot_coeff_tr_ep_rs(mean_rewards=mean_rewards, coeff=alphas)
