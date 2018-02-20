import tensorflow as tf
from keras.layers import Dense

__all__ = ['fully_connected',
           'one_hot_encoding']

def fully_connected(input, output_dim, activation_fn=None, scope=None):
    with tf.variable_scope(scope):
        w = tf.get_variable(name='w',
                                 shape=[input.get_shape()[1], output_dim],
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer())
        b = tf.get_variable(name='b',
                                 shape=[output_dim],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer())
        if activation_fn is None:
            y = tf.matmul(input, w) + b
        else:
            y = activation_fn(tf.matmul(input, w) + b)
    return y


def one_hot_encoding():
    raise NotImplementedError


