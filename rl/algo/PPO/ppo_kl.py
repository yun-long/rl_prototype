from rl.tf.models import MLP
from rl.policy.gp_mlp import GaussianPolicy_MLP
from rl.value.value_mlp import ValueMLP
from rl.env.env_list import env_IDs
from rl.sampler.advanced_sampler import AdvancedSampler
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
class PPO(object):

    def __init__(self, policy_mlp, value_mlp, sess=None, epsilon=0.2, c_lrate=1e-4, a_lrate=1e-4):
        self.pol_mlp = policy_mlp
        self.val_mlp = value_mlp
        self.sess = sess or tf.get_default_session()

        # actor and critic
        self._define_critic(c_lrate)
        self._define_actor(beta=0.5, a_lrate=a_lrate)

        #

    def _define_critic(self, c_lrate):
        self.target_v = tf.placeholder(tf.float32, shape=[None, 1], name='target_v')
        self.loss_c = tf.losses.mean_squared_error(self.val_mlp.out, self.target_v)
        self.train_opt_c = tf.train.AdamOptimizer(c_lrate).minimize(self.loss_c)

    def _define_actor(self,a_lrate, beta=3.):
        self.target_adv = tf.placeholder(tf.float32, shape=[None, 1], name='target_adv')
        self.old_prob = tf.placeholder(tf.float32, shape=[None, 1], name='old_prob')
        new_prob = self.pol_mlp.log_prob
        ratio = new_prob / self.old_prob
        kl = tf.contrib.distributions.kl(self.old_prob, new_prob)
        self.loss_a = tf.losses.mean_squared_error(ratio * self.target_adv, beta * kl)
        self.train_opt_a = tf.train.AdamOptimizer(a_lrate).minimize(self.loss_a)

    def train_critic(self, states, target_v):
        feed_dict = {self.val_mlp.x: states,
                     self.target_v: target_v}
        return self.sess.run(self.loss_c, feed_dict)

    def train_actor(self, states, actions, target_adv):
        feed_dict = {self.pol_mlp.mlp.x: states,
                     self.pol_mlp.test_action: actions,
                     self.target_adv: target_adv}
        return self.sess.run(self.loss_a, feed_dict)

    def eval_critic(self):
        pass

    def eval_actor(self):
        pass



if __name__ == '__main__':
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)
    #
    env = gym.make(env_IDs.Pendulum)
    s_dims = env.observation_space.shape[0]
    a_dims = env.action_space.shape[0]
    #
    pol_nn_layers = [s_dims, 64, 64, a_dims]
    pol_nn_activ  = [tf.nn.relu, tf.nn.relu, tf.nn.tanh]
    val_nn_layers = [s_dims, 64, 64, 1]
    val_nn_activ  = [tf.nn.relu, tf.nn.relu, tf.nn.tanh]
    env.seed(seed)
    #
    sess = tf.Session()
    pol_fn = GaussianPolicy_MLP(pol_nn_layers, pol_nn_activ, sess)
    val_fn = ValueMLP(val_nn_layers, val_nn_activ, sess)
    ppo = PPO(pol_fn, val_fn, sess=sess)
    sampler = AdvancedSampler(env)
    #
