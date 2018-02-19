"""
Gaussian policy, Fully connected layer policy

Reference: John Schulman, Trust Region Policy Optimization
"""

import tensorflow as tf
import numpy as np

class GPFC(object):
    """
    Gaussian policy for continuous states and actions environments
    """
    def __init__(self, env, learning_rate=1e-2):
        self.n_action = env.action_space.shape[0]
        self.n_state = env.observation_space.shape[0]
        self.learning_rate = learning_rate
        self.model()

    def model(self):
        with tf.variable_scope('policy'):
            # define input layer, input states
            self.state = tf.placeholder(tf.float32, shape=[None, self.n_state], name="states")
            # define hidden layers
            with tf.variable_scope('l1'):
                w1 = tf.get_variable(name="w1",
                                     shape=[self.n_state, 30],
                                     dtype=tf.float32,
                                     initializer=tf.random_normal_initializer())
                b1 = tf.get_variable(name="b1",
                                     shape=[30],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer())
                h0 = tf.nn.relu(tf.matmul(self.state, w1) + b1)
            with tf.variable_scope('l2'):
                w2 = tf.get_variable(name="w2",
                                     shape=[30, self.n_action],
                                     dtype=tf.float32,
                                     initializer=tf.random_normal_initializer())
                b2 = tf.get_variable(name="b2",
                                     shape=[self.n_action],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer())
                self.mu = tf.matmul(h0, w2) + b2 # output of the neural nets, namely, the predicted actions
                self.mu = tf.squeeze(self.mu)
            # the true actions.
            self.sigma = tf.constant(1e1, dtype=tf.float32, shape=[self.n_action])
            self.norm_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)

            self.pred_action = self.norm_dist.sample([1])
            #
            # TODO:
            self.target_action = tf.placeholder(tf.float32, shape=[None, self.n_action], name="actions")
            self.target_adv = tf.placeholder(tf.float32, shape=[None, 1], name="advantages")
            #
            with tf.variable_scope('loss'):
                # TODO:
                self.loss = -self.norm_dist.log_prob(self.target_action) * self.target_adv

            with tf.variable_scope('optimizer'):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                self.train_op = self.optimizer.minimize(loss=self.loss,
                                                        global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        state = np.reshape(state, (-1, state.shape[0]))
        sess = sess or tf.get_default_session()
        action = sess.run([self.pred_action], {self.state: state})[0]
        return action[0]

    def train(self, state, action, adv, sess=None):
        # TODO: not working
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state,
                     self.target_action: action,
                     self.target_adv: adv}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


