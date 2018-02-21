from rl.tf.models import MLP


class ValueMLP(object):

    def __init__(self, sizes, activations, sess=None):
        self.mlp = MLP(sizes, activations)
        self.sess = sess

    def predict(self, states):
        value = self.sess.run(self.mlp.out, {self.mlp.x: states})
        return value

