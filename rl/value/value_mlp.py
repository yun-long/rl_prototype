from rl.tf.models import MLP
import tensorflow as tf

class ValueMLP(object):

    def __init__(self, sizes, activations, sess=None):
        self.sess = sess
        # input
        self.states = tf.placeholder(tf.float32, shape=[None, sizes[0]], name='states')
        # target v
        self.target_v = tf.placeholder(tf.float32, shape=[None, 1], name='target_v')
        # hidden layer and output
        self.mlp = MLP(self.states, sizes, 'value', True, activations)

    def predict(self, states):
        value = self.sess.run(self.mlp.out, {self.states: states})
        return value

