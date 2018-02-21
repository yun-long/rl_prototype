from rl.tf.models import MLP
from rl.policy.gp_mlp import GaussianPolicy_MLP
from rl.value.value_mlp import ValueMLP
from rl.env.env_list import env_IDs
#
import gym
import tensorflow as tf
import numpy as np

# choose PPO version
obj = ['clipping', 'kl_penalty']
#
opts = {'obj_ID': 0, # choose the surrogate objectives
        'n_episod': 200,
        'n_trans': 3000,
        'n_epochs': 15,
        'sigma': 1,
        'discount': 0.99,
        'lam_trace': 0.95,
        'epsilon': 0.2,
        'bathch_size': 32}
#
env = gym.make(env_IDs.Pendulum)
s_dims = env.observation_space.shape[0]
a_dims = env.action_space.shape[0]
#
pol_nn_layers = [s_dims, 64, 64, a_dims]
pol_nn_activ  = [tf.nn.relu, tf.nn.relu, tf.nn.tanh]
val_nn_layers = [s_dims, 64, 64, 1]
val_nn_activ  = [tf.nn.relu, tf.nn.relu, tf.nn.tanh]
#
class PPO(object):

    def __init__(self, policy, value, sess):
        pass

    def train_v(self):
        pass

    def train_pol(self):
        pass

    def eval_v(self):
        pass

    def eval_pol(self):



if __name__ == '__main__':
    seed = 123
    np.random.seed(seed)
    env.seed(seed)
    tf.set_random_seed(seed)
    #
    sess = tf.Session()
    policy_fn = GaussianPolicy_MLP(pol_nn_layers, pol_nn_activ, sess)
    value_fn = ValueMLP(val_nn_layers, val_nn_activ, sess)
    ppo = PPO(policy_fn, value_fn, sess)
    #
