import tensorflow as tf
import ot
import numpy as np
#

def alpha_fn(alpha=1.0):
    if alpha == 1.0: # kl divergence
        f = lambda x: x * tf.log(x) - (x - 1)
    elif alpha == 0.0:
        f = lambda x: -tf.log(x) + (x - 1)
    elif alpha == 'gan' or alpha == 'GAN':
        f = lambda x: x * tf.log(x) - (1 + x) * tf.log((x + 1) / 2)
    else:
        f = lambda x: ((tf.pow(x, alpha) - 1) - alpha * (x - 1)) / (alpha * (alpha - 1))
    return f

def f_div(f, log_p, log_q):
    ratio = tf.exp(log_p - log_q)
    dist = tf.reduce_sum(tf.exp(log_q) * f(ratio))
    return dist

def w2(action, p, q):
    M2 = ot.dist(x1=action, x2=action, metric='sqeuclidean')
    M2 /= M2.max()
    reg = 1e-2
    dist = ot.sinkhorn2(p, q, M2, reg)
    return dist


def build_mlp(input, sizes, activations, trainable):
    last_out = input
    for l, size in enumerate(sizes):
        last_out = tf.layers.dense(inputs=last_out,
                                   units=size,
                                   activation=activations[l],
                                   kernel_initializer=tf.glorot_uniform_initializer(),
                                   trainable=trainable,
                                   name='fc{}'.format(l+1))
    return last_out

