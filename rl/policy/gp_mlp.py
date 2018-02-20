import tensorflow as tf
import numpy as np
from rl.tf.models import MLP

class GaussianPolicy_MLP(object):

    def __init__(self, sess, sizes, activations=None, init_sigma=1.):
        self.mlp = MLP(sizes, activations)
        self.sess = sess

        # action tensor (diagonal Gaussian)
        act_dim = sizes[-1]
        self.logsigs = tf.Variable(np.log(init_sigma) * tf.ones(([1, act_dim])))
        self.sigs = tf.exp(self.logsigs)
        self.norm_dist = tf.contrib.distributions.MultivariateNormalDiag(self.mlp.out,
                                                                         self.sigs)
        self.act_tensor = self.norm_dist.sample()

        # action probability
        self.test_action = tf.placeholder(tf.float32, shape=[None, act_dim])
        self.log_prob = tf.expand_dims(self.norm_dist.log_prob(self.test_action), axis=-1)

        # policy entropy (for logging only)
        self.entropy = tf.reduce_sum(self.logsigs) + act_dim * np.log(2 * np.pi * np.e) / 2

    def predict(self, state):
        return np.squeeze(self.sess.run(self.act_tensor, {self.mlp.x : np.asmatrix(state)}), axis=0)

    def get_log_proba(self, state, action):
        return self.sess.run(self.log_prob, {self.mlp.x: state, self.test_action: action})