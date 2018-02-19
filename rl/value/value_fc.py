import tensorflow as tf
import numpy as np

class ValueFC(object):
    """
    Value function approximator for continous states environments
    """
    def __init__(self, env, learning_rate=1e-2):
        self.n_state = env.observation_space.shape[0]
        self.learning_rate = learning_rate
        self.model()

    def model(self):
        with tf.variable_scope('v'):
            self.state = tf.placeholder(tf.float32, shape=[None, self.n_state], name="states")
            # define hidden layers
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1',
                                     shape=[self.n_state, 30],
                                     dtype=tf.float32,
                                     initializer=tf.random_normal_initializer())
                b1 = tf.get_variable(name='b1',
                                     shape=[30],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer())
                h0 = tf.nn.relu(tf.matmul(self.state, w1)+b1)
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2',
                                     shape=[30, 1],
                                     initializer=tf.random_normal_initializer())
                b2 = tf.get_variable('b2',
                                     shape=[1],
                                     initializer=tf.constant_initializer())
                self.val = tf.matmul(h0, w2) + b2
                # self.val = tf.squeeze(self.val)
            #
            self.target_val = tf.placeholder(tf.float32, shape=[None, 1], name='advantages')
            with tf.variable_scope('loss'):
                self.loss = tf.squared_difference(self.target_val, self.val)

            with tf.variable_scope('optimizer'):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                self.train_opt = self.optimizer.minimize(self.loss,
                                                         global_step=tf.contrib.framework.get_global_step())


    def predict(self, state, sess=None):
        state = np.reshape(state, (-1, state.shape[0]))
        sess = sess or tf.get_default_session()
        val = sess.run([self.val], {self.state: state})
        return val[0]

    def train(self, state, val, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state,
                     self.target_val: val}
        _, loss = sess.run([self.train_opt, self.loss], feed_dict)
        return loss