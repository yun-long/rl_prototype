import tensorflow as tf
import os
import ot
import numpy as np
import pandas as pd
#
from rl.misc.utilies import get_dirs
#
env_ID = "Pendulum-v0"
#
path_all_results = os.path.join(os.path.realpath("../../../"), 'results')
path_ppo_results = get_dirs(os.path.join(path_all_results, 'ppo'))
path_env_result = get_dirs(os.path.join(path_ppo_results, env_ID))
path_csv = os.path.join(path_env_result, 'data.csv')
#
columns = ['methods', 'alphas', 'trials', 'episodes', 'rewards', 'losses_c', 'losses_a', 'divergences', 'entropies', "beta"]
#
data = pd.DataFrame(columns=columns)
seed = 12345
#
params = {'methods': {'clip': [None],
                      # 'f': [1.0, 2.0, 'GAN'],
                      # 'w2': [None]
                      },
          'num_trials': 5,
          'num_episodes': 100,
          'num_sample_trans': 3200,
          'epochs': 10,
          'batch_size': 32,
          'gamma': 0.99,
          'lam': 0.95,
          'clip_param': 0.2,
          'beta': 3.0,
          'ent_coeff': 0.0,
          'c_lrate': 3e-4,
          'a_lrate': 3e-4,
          'kl_target': 0.01,
          'w2_target': 0.1}