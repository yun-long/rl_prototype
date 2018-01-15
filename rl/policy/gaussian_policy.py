import tensorflow as tf


# Step_based
class GaussianPolicy(object):
    """
    Policy Function Approximator
    """
    def __init__(self, env, featurizer, learning_rate, scope="policy_estimator"):
        self.input_dim = featurizer.num_features
        self.output_dim = env.action_space.shape[0]
        with tf.variable_scope(scope):
            #
            self.state = tf.placeholder(dtype=tf.float32, shape=[self.input_dim], name="state_features")
            # target is the TD Error
            self.target = tf.placeholder(dtype=tf.float32, name="target")
            #
            self.mu = tf.contrib.layers.fully_connected(inputs=tf.expand_dims(self.state, 0),
                                                        num_outputs=1,
                                                        activation_fn=None)
            self.mu = tf.squeeze(self.mu)
            #
            self.sigma = tf.contrib.layers.fully_connected(inputs=tf.expand_dims(self.state, 0),
                                                           num_outputs=1,
                                                           activation_fn=None)
            self.sigma = tf.squeeze(self.sigma)
            self.sigma = tf.nn.softplus(self.sigma) + 1e-5
            #
            self.norm_distribution = tf.contrib.distributions.Normal(self.mu, self.sigma)
            self.action = self.norm_distribution._sample_n(1)
            self.action = tf.clip_by_value(self.action, env.action_space.low[0], env.action_space.high[0])
            # self.action = tf.squeeze(self.action)
            #
            self.loss = -self.norm_distribution.log_prob(self.action) * self.target
            #
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(loss=self.loss,
                                                    global_step=tf.contrib.framework.get_global_step())
        self.rbf_featurizer = featurizer

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        featurized_state = self.rbf_featurizer.transform(state)
        return sess.run(self.action, {self.state: featurized_state})

    def update(self, state, action, target, sess=None):
        sess = sess or tf.get_default_session()
        featurized_state = self.rbf_featurizer.transform(state)
        feed_dict = {self.state: featurized_state,
                     self.action: action,
                     self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

