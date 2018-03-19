from rl.algo.freps.constant import results_path, freps_path, get_dirs
from rl.algo.freps.freps import freps
from rl.featurizer.one_hot_featurizer import OneHotFeaturizer
from rl.sampler.standard_sampler import StandardSampler
from rl.value.value_estimator import ValueEstimator
from rl.policy.discrete_policy import DistributionPolicy_V1, DistributionPolicy
#
from gym.envs.algorithmic.copy_ import CopyEnv
from gym.envs.algorithmic.duplicated_input import DuplicatedInputEnv
from gym.spaces.tuple_space import Tuple
import numpy as np
import pandas as pd
import gym
import os
import sys
import datetime
#
methods = ['freps', 'freps_light']
env_IDs = ['Copy-v0', 'DuplicatedInput-v0', 'RepeatCopy-v0', 'Reverse-v0']
alphas = [-15.0, -10.0,  -2.0,  0.0,  1.0, 2.0]
if __name__ == '__main__':
    method = methods[1]
    #
<<<<<<< HEAD
    env_ID = env_IDs[0]
=======
    env_ID = env_IDs[1]
    # env = gym.make(env_ID)
    n_actions = [20, 40]
>>>>>>> cabb5a3f9cf46cfe9538542e880731d312e3e696
    env_path = get_dirs(os.path.join(freps_path, env_ID))
    # env = gym.make(env_ID)
    n_bases = [5, 10]
    # pandas data frame for saving the results.
    data_name = method
    columns = ['alpha', 'trial', 'episode', 'reward']
<<<<<<< HEAD
=======
    # value featurizer and sampler
>>>>>>> cabb5a3f9cf46cfe9538542e880731d312e3e696
    # parameters
    num_episodes = 100
    num_trials = 10
    num_samples = 1000
<<<<<<< HEAD
    df_data = pd.DataFrame(columns=columns)
    for i_b, base in enumerate(n_bases):
        # env = DuplicatedInputEnv(base=base)
        env = CopyEnv(base=base)
        print("Action Space: ", env.action_space)
        print("Observation Space: ", env.observation_space)
        # value featurizer and sampler
        val_faeturizer = OneHotFeaturizer(env)
        sampler = StandardSampler(env)
        data_name = method + '_' + str(base)
=======
    for i_A, n_action in enumerate(n_actions):
        # env = CopyEnv(base=int(n_action / 4))
        env = gym.make(env_ID)
        print("Action Space: ", env.action_space)
        print("Observation Space: ", env.observation_space)
        data_name = method + '_' + str(n_action)
        df_data = pd.DataFrame(columns=columns)
        val_faeturizer = OneHotFeaturizer(env)
        sampler = StandardSampler(env)
>>>>>>> cabb5a3f9cf46cfe9538542e880731d312e3e696
        for i_alpha, alpha in enumerate(alphas):
            data_name = data_name + '_' + str(alpha)
            seed = 123456
            env.seed(seed=seed)
            rnd = np.random.RandomState(seed=seed)
            for i_trial in range(num_trials):
                val_fn = ValueEstimator(featurizer=val_faeturizer)
                # pol_fn = DistributionPolicy_V1(env=env, rnd = rnd)
                pol_fn = DistributionPolicy_V1(env=env, rnd=rnd)

                state = env.reset()
                #
                ep_rewards = freps(i_trial, val_fn=val_fn, pol_fn=pol_fn,
                                   sampler=sampler, num_ep=num_episodes,
                                   num_sp=num_samples, alpha=alpha, rnd=rnd, columns=columns)
                df_data = df_data.append(ep_rewards, ignore_index=True)
        env_path = get_dirs(os.path.join(freps_path, env_ID))
        env_run_path = get_dirs(os.path.join(env_path, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
        data_path = os.path.join(env_run_path, data_name + '_data.csv')
        df_data.to_csv(data_path, index=False)
        print("Data saved at: ".format(data_path))


