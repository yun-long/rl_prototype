from rl.featurizer.one_hot_featurizer import OneHotFeaturizer
from rl.policy.value_estimator import ValueEstimator
from rl.sampler.standard_sampler import StandardSampler
from rl.misc.dual_function import *
from rl.misc.plot_rewards import plot_trail_episode_rewards
from rl.misc.plot_value import plot_2D_value
from rl.policy.discrete_policy import DistributionPolicy
#
from gym.envs.toy_text.nchain import NChainEnv
import numpy as np

def reps_step_based(val_featurizer, policy, sampler, num_episodes, param_eta0, param_v0, epsilon):
    episodes_rewards = []
    #
    for i_episode in range(num_episodes):
        data_set, mean_reward = sampler.sample_data(policy=policy, N=800)
        # process sampled data
        rewards, features_diff, sa_n, keys = sampler.count_data(data_set, val_featurizer)
        # optimize dual function
        param_eta, param_v, A, g = optimize_dual_fn_params(rewards=rewards,
                                                             features_diff=features_diff,
                                                             init_eta=param_eta0,
                                                             init_v=param_v0,
                                                             epsilon=epsilon,
                                                             sa_n=sa_n)
        policy.update(A=A,param_eta=param_eta, param_v=param_v, g=g, keys=keys)
        value_fn.update(new_param_v=param_v)
        episodes_rewards.append(mean_reward)
        print("\rEpisode {}, Expected Return {}.".format(i_episode, mean_reward))
    return episodes_rewards

if __name__ == '__main__':
    #
    env = NChainEnv(n=5)
    print("Action Space : ", env.action_space)
    print("Observation Space : ", env.observation_space)
    # define featurizer for value function
    val_featurizer = OneHotFeaturizer(env=env)
    # define sampler
    sampler = StandardSampler(env)
    #
    # initialization paramteres for dual function
    epsilon = 0.1
    #
    num_trails = 10
    num_episodes = 50
    trail_rewards = np.zeros((num_trails, num_episodes))
    for i_trail in range(num_trails):
        # define value function
        value_fn = ValueEstimator(featurizer=val_featurizer)
        # define policy
        policy = DistributionPolicy(env)
        param_eta0 = 5.0
        param_v0 = value_fn.param_v0
        #
        reward = reps_step_based(val_featurizer=val_featurizer,
                                 policy = policy,
                                 sampler=sampler,
                                 num_episodes=num_episodes,
                                 param_eta0=param_eta0,
                                 param_v0=param_v0,
                                 epsilon=epsilon)
        trail_rewards[i_trail, :] = reward
    #
    plot_2D_value(env, value_fn, show=True)
    #
    plot_trail_episode_rewards(trail_rewards, show=False)
