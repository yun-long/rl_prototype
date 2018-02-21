import tensorflow as tf

__all__ = ['MLP']

class MLP(object):
    """
    Multilayer perceptron
    """
    def __init__(self, input, sizes, name, trainable, activations=None):
        if activations is None:
            activations = [tf.nn.relu] * (len(sizes) - 2) + [tf.identity]
        last_out = input
        with tf.variable_scope(name):
            for l, size in enumerate(sizes[1:]):
                last_out = tf.layers.dense(inputs=last_out,
                                           units=size,
                                           activation=activations[l],
                                           kernel_initializer=tf.glorot_uniform_initializer(),
                                           trainable=trainable)
        self.out = last_out
        self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
