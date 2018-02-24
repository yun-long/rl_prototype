from rl.tf.models import MLP
import tensorflow as tf

class ValueMLP(object):

    def __init__(self, states, target_v, sizes, activations, sess=None):
        self.sess = sess
        # input
        # self.states = states
        # target v
        # self.target_v = target_v
        # hidden layer and output
        self.mlp = MLP(states, sizes, True, activations)

    def predict(self, states):
        value = self.sess.run(self.mlp.out, {self.states: states})
        return value

