import tensorflow as tf
import numpy as np
from rl.tf.models import MLP

class GaussianPolicy_MLP(object):

    def __init__(self, sizes, activations=None, sess = None, init_sigma=1.):
        self.sess = sess or tf.get_default_session()
        # input
        self.states = tf.placeholder(tf.float32, shape=[None, sizes[0]], name='states')
        self.actions = tf.placeholder(tf.float32, shape=[None, sizes[-1]])
        self.target_adv = tf.placeholder(tf.float32, shape=[None, 1], name='target_adv')
        # action tensor (diagonal Gaussian)
        act_dim = sizes[-1]
        self.logsigs = tf.Variable(np.log(init_sigma) * tf.ones(([1, act_dim])))
        self.sigs = tf.exp(self.logsigs)
        # hidden layer and output
        self.mlp = MLP(input=self.states, sizes=sizes, name='pi', trainable=True, activations= activations)
        self.norm_dist = tf.contrib.distributions.MultivariateNormalDiag(self.mlp.out,
                                                                         self.sigs)
        #
        self.old_mlp = MLP(input=self.states, sizes=sizes, name='old_pi', trainable=False, activations=activations)
        self.old_norm_dist = tf.contrib.distributions.MultivariateNormalDiag(self.mlp.out,
                                                                             self.sigs)
        self.pred_action = self.norm_dist.sample()

        # action probability
        self.log_prob = tf.expand_dims(self.norm_dist.log_prob(self.actions), axis=-1)
        self.old_log_prob = tf.expand_dims(self.old_norm_dist.log_prob(self.actions), axis=-1)

        # policy entropy (for logging only)
        self.entropy = tf.reduce_sum(self.logsigs) + act_dim * np.log(2 * np.pi * np.e) / 2

    def predict(self, state):
        return np.squeeze(self.sess.run(self.pred_action, {self.states : np.asmatrix(state)}), axis=0)

    def get_log_proba(self, state, action):
        return self.sess.run(self.log_prob, {self.states: state, self.actions: action})

    def get_mean(self, state, action):
        return self.sess.run(self.norm_dist, {self.states: state, self.actions: action})