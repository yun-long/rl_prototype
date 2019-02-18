"""
An implementation of REPS on Random Jumpy Environment, Gaussian Policy, Linear mean
"""
from rl.policy.gp_linear_mean import GPLinearMean
from rl.featurizer.non_featurizer import NoneFeaturizer
from rl.featurizer.poly_featurizer import PolyFeaturizer
from rl.value.value_estimator import ValueEstimator
from rl.misc.dual_function import optimize_dual_fn
from rl.sampler.standard_sampler import StandardSampler
from rl.env.random_jump import RandomJumpEnv
from rl.misc.plot_rewards import plot_tr_ep_rs
#
#
import numpy as np
#
# env = Continuous_MountainCarEnv()
env = PendulumEnv()
# env = RandomJumpEnv()

#
print("Action space : ", env.action_space)
print("Action space low : ", env.action_space.low)
print("Action space high: ", env.action_space.high)
print("Observation space : ", env.observation_space)
print("Observation space low : ", env.observation_space.low)
print("Observation space high: ", env.observation_space.high)
#
pol_featurizer = NoneFeaturizer(env)
#
val_featurizer = PolyFeaturizer(env, degree=2)
#
sampler = StandardSampler(env)
# Number of episodes
num_episodes = 50
num_trails = 10
num_samples = 500
#
epsilon = 1.0
#
mean_rewards = np.zeros((num_trails, num_episodes))
for i_trail in range(num_trails):
    rnd = np.random.RandomState(seed=43225801)
    #
    policy = GPLinearMean(env, pol_featurizer)
    value = ValueEstimator(val_featurizer)
    #
    v0 = np.squeeze(value.param_v)
    eta0 = 10.0
    for i_episode in range(num_episodes):
        #
        data_set, mean_rs = sampler.sample_data(policy=policy,N=num_samples)
        #
        rewards, val_feat_diff, A, Phi = sampler.process_data(data_set, pol_featurizer, val_featurizer)
        #
        eta, v, weights = optimize_dual_fn(rewards=rewards, features_diff=val_feat_diff,
                                                 init_eta=eta0, epsilon=epsilon, rnd=rnd)
        # Update policy
        policy.update_wml(Weights=weights, Phi=Phi, A=A)
        # update value function
        value.update(new_param_v=v)
        print("Trails : {}, Episode : {}, Reward: {}".format(i_trail, i_episode, mean_rs))
        mean_rewards[i_trail, i_episode] = np.mean(rewards)
#
plot_tr_ep_rs(mean_rewards,show=True)


