"""
Reference : https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/12_Proximal_Policy_Optimization/simply_PPO.py
"""
import tensorflow as tf
import numpy as np


class GPFC(object):

    def __init__(self, env, learning_rate):
        self.n_state = env.observation_space.shape[0]
        self.n_action = env.action_space.shape[0]
        self.learning_rate_a = 0.0001
        self.learning_rate_c = 0.0002
        self.epsilon = 0.2
        self.method = 'clip'
        self.A_update_step = 10
        self.C_update_step = 10
        #
        self.states = tf.placeholder(tf.float32, shape=[None, self.n_state], name='states')
        # build_critic
        self._critic()
        # build actor
        self._actor()
        #

    def _critic(self, scope='critic'):
        with tf.variable_scope(scope):
            L1 = tf.layers.dense(inputs=self.states,
                                 units=100,
                                 activation=tf.nn.relu,
                                 name=scope)
            self.v = tf.layers.dense(L1, 1)
            self.rewards = tf.placeholder(tf.float32, [None, 1], name='rewards')
            with tf.variable_scope('loss'):
                self.adv = self.rewards - self.v
                self.loss_c = tf.reduce_mean(tf.square(self.rewards - self.v))
            with tf.variable_scope('train_opt'):
                self.train_opt_c = tf.train.AdamOptimizer(learning_rate=self.learning_rate_c).minimize(self.loss_c)

    def _actor(self, scope='actor'):
        with tf.variable_scope(scope):
            pi, pi_params = self._build_mode(scope='pi', trainable=True)
            old_pi, old_pi_params = self._build_mode(scope='old_pi', trainable=False)
            self.pred_action = tf.squeeze(pi.sample(1), axis=0)
            with tf.variable_scope('update_old_pi'):
                self.update_old_pi = [old_params.assign(params) for params, old_params in zip(pi_params, old_pi_params)]
            self.target_actions = tf.placeholder(tf.float32, shape=[None, self.n_action], name='target_actions')
            self.target_adv = tf.placeholder(tf.float32, shape=[None, 1], name='target_adv')
            # the essential part of PPO algorithm
            with tf.variable_scope('surrogate'):
                ratio = pi.prob(self.target_actions) / old_pi.prob(self.target_actions)
                surr_fn = ratio * self.target_adv
            with tf.variable_scope('loss'):
                if self.method == 'clip':
                    self.loss_a = - tf.reduce_mean(tf.minimum(
                        surr_fn,
                        tf.clip_by_value(ratio, 1.-self.epsilon, 1.+self.epsilon)*self.target_adv))
            with tf.variable_scope('train_opt'):
                self.train_opt_a = tf.train.AdamOptimizer(learning_rate=self.learning_rate_a).minimize(self.loss_a)



    def _build_mode(self, scope=None, trainable=True):
        with tf.variable_scope(scope):
            # hidden layer
            L1 = tf.layers.dense(inputs=self.states,
                                 units=100,
                                 activation=tf.nn.relu,
                                 trainable=trainable)
            mu = 2 * tf.layers.dense(inputs=L1,
                                     units=self.n_action,
                                     activation=None,
                                     trainable=trainable)
            sigma = tf.layers.dense(inputs=L1,
                                    units=self.n_action,
                                    activation=tf.nn.softplus,
                                    trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        return norm_dist, params

    def update(self, states, actions, rewards, sess=None):
        sess = sess or tf.get_default_session()
        sess.run(self.update_old_pi)
        adv = sess.run(self.adv, {self.states: states,
                                       self.rewards: rewards})
        # update actor:
        feed_dict_a = {self.states: states,
                       self.target_actions: actions,
                       self.target_adv: adv}
        if self.method == 'clip':
            [sess.run(self.train_opt_a, feed_dict_a) for _ in range(self.A_update_step)]
        # update critic:
        feed_dict_c = {self.states: states,
                       self.rewards: rewards}
        [sess.run(self.train_opt_c, feed_dict_c) for _ in range(self.C_update_step)]

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        # state = state[np.newaxis, :]
        state = np.reshape(state, (-1, state.shape[0]))
        a = sess.run(self.pred_action, {self.states: state})[0]
        return np.clip(a, -2, 2)

    def value(self, state, sess= None):
        sess = sess or tf.get_default_session()
        # state = state[np.newaxis, :]
        state = np.reshape(state, (-1, state.shape[0]))
        v = sess.run(self.v, {self.states: state})
        return v

