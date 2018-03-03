import tensorflow as tf
import numpy as np
import gym


class GpMlp(object):
    def __init__(self, sess, env):
        self.sess = sess
        self.a_dim = env.action_space.shape[0]
        self.s_dim = env.observation_space.shape[0]
        self.hidden_sizes = [64, 64, self.a_dim]
        self.act_layers = [tf.nn.relu, tf.nn.relu, tf.nn.tanh]
        #
        with tf.variable_scope('actor'):
            self.states = tf.placeholder(tf.float32, shape=[None, self.s_dim], name='states')
            self.actions = tf.placeholder(tf.float32, shape=[None, self.a_dim], name='actions')
            self.advantages = tf.placeholder(tf.float32, shape=[None, 1], name='advantages')
            self.mu = self.__mlp(self.states, 'fc')


    def __mlp(self, input, name):
        with tf.variable_scope(name):
            last_out = input
            for l, size in enumerate(self.hidden_sizes):
                last_out = tf.layers.dense(last_out, size,
                                           activation=self.act_layers[l],
                                           kernel_initializer=tf.glorot_uniform_initializer(),
                                           name='fc{0}'.format(l+1))
        return last_out

    def predict_action(self):
        pass


class ValueMlp(object):
    def __init__(self, sess, env):
        self.sess = sess
        self.a_dim = env.action_space.shape[0]
        self.s_dim = env.observation_space.shape[0]
        self.hidden_sizes = [64, 64, 1]
        self.act_layers = [tf.nn.relu, tf.nn.relu, tf.identity]
        #
        with tf.variable_scope('critic'):
            self.states = tf.placeholder(tf.float32, shape=[None, self.s_dim], name='states')
            self.target_v = tf.placeholder(tf.float32, shape=[None, 1], name='target_v')
            self.pred_v = self.__mlp(self.states, 'fc')

    def __mlp(self, input, name):
        with tf.variable_scope(name):
            last_out = input
            for l, size in enumerate(self.hidden_sizes):
                last_out = tf.layers.dense(last_out, size,
                                           activation=self.act_layers[l],
                                           kernel_initializer=tf.glorot_uniform_initializer(),
                                           name='fc{0}'.format(l+1))
        return last_out

class PPO(object):
    def __init__(self):
        pass


if __name__ == '__main__':
    #
    env = gym.make("Pendulum-v0")
    with tf.Session() as sess:
        policy = GpMlp(sess=sess, env=env)
        vaue = ValueMlp(sess=sess, env=env)
        sess.run(tf.global_variables_initializer())
        tf.summary.FileWriter('log', sess.graph)

