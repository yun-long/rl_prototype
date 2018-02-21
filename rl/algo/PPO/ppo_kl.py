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
    """
    Proximal Policy Optimization
    """

    def __init__(self, policy, value, sess=None, epsilon=0.2, c_lrate=1e-4, a_lrate=1e-4):
        self.sess = sess or tf.get_default_session()

        # actor and critic
        self._define_critic(value, c_lrate)
        self._define_actor(policy, a_lrate=a_lrate, beta=3.)

        #

    def _define_critic(self, value, c_lrate):
        self.val_fn = value
        self.loss_c = tf.losses.mean_squared_error(self.val_fn.mlp.out, self.val_fn.target_v)
        self.train_opt_c = tf.train.AdamOptimizer(c_lrate).minimize(self.loss_c)

    def _define_actor(self, policy, a_lrate, beta=3.):
        self.pol_fn = policy
        #
        ratio = self.pol_fn.log_prob / self.pol_fn.old_log_prob
        kl = tf.contrib.distributions.kl_divergence(self.pol_fn.old_norm_dist, self.pol_fn.norm_dist)
        self.kl_mean = tf.reduce_mean(kl)
        self.loss_a = - tf.reduce_mean(ratio * self.pol_fn.target_adv - beta * kl)
        self.train_opt_a = tf.train.AdamOptimizer(a_lrate).minimize(self.loss_a)

    def update_old_pi(self):
        for oldp, p in zip(self.pol_fn.old_mlp.params, self.pol_fn.mlp.params):
            oldp.assign(p)

    def train_critic(self, states, target_v):
        feed_dict = {self.val_fn.states: states,
                     self.val_fn.target_v: target_v}
        return self.sess.run(self.train_opt_c, feed_dict)

    def train_actor(self, states, actions, target_adv):
        feed_dict = {self.pol_fn.states: states,
                     self.pol_fn.actions: actions,
                     self.pol_fn.target_adv: target_adv}
        return self.sess.run(self.train_opt_a, feed_dict)

    def eval_critic(self, states):
        feed_dict = {self.val_fn.states: states}
        return self.sess.run(self.loss_c, feed_dict)

    def eval_actor(self, states, action):
        feed_dict = {self.pol_fn.states: states,
                     self.pol_fn.actions: action}
        return self.sess.run(self.loss_a, feed_dict)

def next_batch_idx(batch_size, data_set_size):
    batch_idx_list = np.random.choice(data_set_size, data_set_size, replace=False)
    for batch_start in range(0, data_set_size, batch_size):
        yield batch_idx_list[batch_start: min(batch_start+batch_size, data_set_size)]


if __name__ == '__main__':
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)
    #
    env = gym.make(env_IDs.Pendulum)
    s_dims = env.observation_space.shape[0]
    a_dims = env.action_space.shape[0]
    env.seed(seed)
    #
    sess = tf.Session()
    # define policy function, multiplelayers perceptron
    pol_nn_layers = [s_dims, 64, 64, a_dims]
    pol_nn_activ  = [tf.nn.relu, tf.nn.relu, tf.nn.tanh]
    pol_fn = GaussianPolicy_MLP(pol_nn_layers, pol_nn_activ, sess)
    # define value function, multiplelayers perceptron
    val_nn_layers = [s_dims, 64, 64, 1]
    val_nn_activ  = [tf.nn.relu, tf.nn.relu, tf.nn.tanh]
    val_fn = ValueMLP(val_nn_layers, val_nn_activ, sess)
    # define Proximal policy optimization class
    ppo = PPO(pol_fn, val_fn, a_dims, sess=sess)
    # define sampler
    sampler = AdvancedSampler(env)
    # initilization
    sess.run(tf.global_variables_initializer())
    #
    for i_episode in range(opts['n_episod']):
        print("-------- episode: {}------------".format(i_episode))
        # sample data
        paths = sampler.rollous(pol_fn, n_trans=opts['n_trans'])
        print("Mean reward: {}".format(np.mean(paths['reward'])))

        # update critic
        #
        for epoch in range(opts['n_epochs']):
            # generate advantages
            adv, values = sampler.get_adv(paths, val_fn, discount=opts['discount'], lam=opts['lam_trace'])
            v_targes = values + adv
            #
            for batch_idx in next_batch_idx(opts['bathch_size'], len(v_targes)):
                ppo.train_critic(paths['state'][batch_idx], v_targes[batch_idx])

        # update actor
        adv, _ = sampler.get_adv(paths, val_fn, discount=opts['discount'], lam=opts['lam_trace'])
        adv = adv / np.std(adv)
        ppo.update_old_pi()
        for epoch in range(opts['n_epochs']):
            for batch_idx in next_batch_idx(opts['bathch_size'], len(adv)):
                ppo.train_actor(paths['state'][batch_idx],
                                paths['action'][batch_idx],
                                adv[batch_idx])


