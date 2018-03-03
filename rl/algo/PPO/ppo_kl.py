import tensorflow as tf
import numpy as np
import gym
from rl.tf.baseline.distributions import make_pdtype


class MlpPolicy(object):

    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, act_space, obs_space, hid_sizes, num_hid_layers):
        a_dim = act_space.shape[0]
        s_dim = obs_space.shape[0]
        self.pdtype = pdtype = make_pdtype(act_space)
        #
        self.state = tf.placeholder(tf.float32, shape=[None, s_dim], name="state")
        #
        with tf.variable_scope('actor'):
            lastout = self.__mlp(self.state, hid_sizes, num_hid_layers)
            mean = tf.layers.dense(inputs=lastout, units=a_dim, name='mu', kernel_initializer=tf.glorot_uniform_initializer())
            logstd = tf.get_variable(name='logstd', shape=[1, a_dim], initializer=tf.ones_initializer)
            pdparam = tf.concat([mean, logstd], axis=1)

        self.pd = pdtype.pdfromflat(pdparam)

        with tf.variable_scope('critic'):
            lastout = self.__mlp(self.state, hid_sizes, num_hid_layers)
            self.vpred = tf.layers.dense(inputs=lastout, units=1, name='value', kernel_initializer=tf.glorot_uniform_initializer())[:, 0]

        self.state_in = []
        self.state_out = []


    @staticmethod
    def __mlp(input, hid_sizes, num_hid_layers):
        lastout = input
        for l in range(num_hid_layers):
            lastout = tf.layers.dense(inputs=lastout,
                                      units=hid_sizes,
                                      activation=tf.nn.tanh,
                                      kernel_initializer=tf.glorot_uniform_initializer(),
                                      trainable=True, name='fc{}'.format(l+1))
        return lastout


    def act(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return np.squeeze(sess.run(self.pd.sample(), feed_dict={self.state: np.asmatrix(state)}), axis=0)

    def value(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.vpred, feed_dict={self.state: np.asmatrix(state)})

    def get_variables(self):
        return tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

    def get_initial_state(self):
        return []


def learn(env, policy_fn, sess, clip_param, entcoeff, gamma, lam):
    #
    act_space = env.action_space
    obs_space = env.observation_space
    #
    pi = policy_fn('pi', act_space, obs_space)
    oldpi = policy_fn('old', act_space, obs_space)
    # some placeholder
    target_adv = tf.placeholder(tf.float32, shape=[None, 1], name='target_adv')
    target_v = tf.placeholder(tf.float32, shape=[None, 1], name='target_v')
    test_action = tf.placeholder(tf.float32, shape=[None, act_space.shape[0]], name='test_action')
    #
    kloldnew = oldpi.pd.kl(other=pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent
    #
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('log', sess.graph)
    #


def policy_fn(name, act_space, obs_space):
    return MlpPolicy(name=name, act_space=act_space, obs_space=obs_space,
                     hid_sizes=64, num_hid_layers=2)

if __name__ == '__main__':
    env = gym.make("Pendulum-v0")
    env.seed(1234)
    #
    with tf.Session() as sess:
        learn(env, policy_fn, sess,
              clip_param=0.2, entcoeff=0.0,
              gamma=0.99, lam=0.95)

