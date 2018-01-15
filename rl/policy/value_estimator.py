import tensorflow as tf

class ValueEstimator(object):

    def __init__(self, env, featurizer, learning_rate, scope="value_estimator"):
        self.input_dim = featurizer.num_features
        self.output_dim = 1
        with tf.variable_scope(scope):
            self.state = tf.placeholder(dtype=tf.float32,
                                        shape=[self.input_dim],
                                        name="state")
            # Target is the true value
            self.target = tf.placeholder(dtype=tf.float32,
                                         name="target")
            self.output_layer = tf.contrib.layers.fully_connected(inputs=tf.expand_dims(self.state, 0),
                                                                  num_outputs=1,
                                                                  activation_fn=None,
                                                                  weights_initializer=tf.zeros_initializer)
            #
            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)
            #
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.loss,
                                                    global_step=tf.contrib.framework.get_global_step())
        self.rbf_featurizer = featurizer

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        featurized_state = self.rbf_featurizer.transform(state)
        return sess.run(self.value_estimate, {self.state: featurized_state})

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        featurized_state = self.rbf_featurizer.transform(state)
        feed_dict = {self.state: featurized_state,
                     self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss

