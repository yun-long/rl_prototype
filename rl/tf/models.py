import tensorflow as tf

__all__ = ['MLP']

class MLP(object):
    """
    Multilayer perceptron
    """
    def __init__(self, sizes, activations=None):
        if activations is None:
            activations = [tf.nn.relu] * (len(sizes) - 2) + [tf.identity]
        self.x = tf.placeholder(tf.float32, shape=[None, sizes[0]])
        last_out = self.x
        for l, size in enumerate(sizes[1:]):
            last_out = tf.layers.dense(inputs=last_out,
                                       units=size,
                                       activation=activations[l],
                                       kernel_initializer=tf.glorot_uniform_initializer())
        self.out = last_out
