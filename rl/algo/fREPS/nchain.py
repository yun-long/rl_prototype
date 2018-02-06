from rl.featurizer.one_hot_featurizer import OneHotFeaturizer
from rl.policy.value_estimator import ValueEstimator
from rl.policy.discrete_policy import DistributionPolicy
from rl.sampler.standard_sampler import StandardSampler
from rl.misc.plot_rewards import plot_tr_ep_rs
from gym.envs.toy_text.nchain import NChainEnv

#
import numpy as np

def f_reps(val_fn, pol_fn, sampler, num_ep, num_sp, init_eta0, init_v0, epsilon):
    ep_rewards = []
    for i_ep in range(num_ep):
        # sample data set from sampler
        data_set, mean_r = sampler.sample_data(policy=pol_fn, N=num_sp)
        #
        rewards, feat_diff, sa_pair_n, sa_pairs = sampler.count_data(data=data_set,
                                                                     featurizer=val_fn.featurizer)
        #
        ep_rewards.append(mean_r)

    return ep_rewards



if __name__ == '__main__':
    env = NChainEnv(n=5)
    print("Action Space : ", env.action_space)
    print("Observation Space : ", env.observation_space)
    #
    val_featurizer = OneHotFeaturizer(env)
    sampler = StandardSampler(env)
    #
    num_episodes = 100
    num_trails = 10
    num_samples = 400
    #
    epsilon = 0.5
    #
    mean_rewards = np.zeros(shape=(num_trails, num_episodes))
    for i_trial in range(num_trails):
        #initialized value funiton and policy function
        val_fn = ValueEstimator(featurizer=val_featurizer)
        pol_fn = DistributionPolicy(env=env)
        # parameters initialization
        eta0 = 10.0
        v0 = val_fn.param_v0
        #
        episodes_rewards = f_reps(val_fn=val_fn,
                             pol_fn=pol_fn,
                             sampler=sampler,
                             num_ep=num_episodes,
                             num_sp=num_samples,
                             init_eta0=eta0,
                             init_v0=v0,
                             epsilon=epsilon)

        mean_rewards[i_trial, :] = episodes_rewards

