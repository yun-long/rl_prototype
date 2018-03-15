from rl.algo.freps.constant import results_path, freps_path, get_dirs
from rl.algo.freps.freps import f_reps
from rl.featurizer.one_hot_featurizer import OneHotFeaturizer
from rl.sampler.standard_sampler import StandardSampler
from rl.value.value_estimator import ValueEstimator
from rl.policy.discrete_policy import DistributionPolicy_V1
#
from gym.envs.algorithmic.copy_ import CopyEnv
from gym.spaces.tuple_space import Tuple
import numpy as np
import pandas as pd
import gym
import os
import sys
#
if __name__ == '__main__':
    #
    env_ID = 'Copy-v0'
    # env = gym.make(env_ID)
    env = CopyEnv(base=5)
    print("Action Space: ", env.action_space)
    print("Observation Space: ", env.observation_space)
    sys.exit()
    env_path = get_dirs(os.path.join(freps_path, env_ID))
    # pandas data frame for saving the results.
    data_path = os.path.join(env_path, 'data.csv')
    columns = ['alpha', 'trial', 'episode', 'reward']
    # value featurizer and sampler
    val_faeturizer = OneHotFeaturizer(env)
    sampler = StandardSampler(env)
    # parameters
    num_episodes = 30
    num_trials = 1
    num_samples = 500
    alphas = [1.0, 2.0]
    for i_alpha, alpha in enumerate(alphas):
        seed = 123456
        env.seed(seed=seed)
        rnd = np.random.RandomState(seed=seed)
        for i_trial in range(num_trials):
            val_fn = ValueEstimator(featurizer=val_faeturizer)
            pol_fn = DistributionPolicy_V1(env=env)

            state = env.reset()
            action = pol_fn.predict_action(state=state)
            print(action)
            #
            # episodes_rewards = f_reps(i_trial, val_fn=val_fn, pol_fn=pol_fn,
            #                           sampler=sampler, num_ep=num_episodes,
            #                           num_sp=num_samples, alpha=alpha, rnd=rnd)

    print(type(env.action_space) is Tuple)


