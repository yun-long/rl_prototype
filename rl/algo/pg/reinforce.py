import numpy as np
import tensorflow as tf
import gym

class REINFORCE(object):

    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.95, outuput_graph=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.beta = 1.0

        self.ep_obs, self.ep_acts, self.ep_rs = [], [], []

        self.object_function()
        self.sess = tf.Session()
        tf.summary.FileWriter(logdir='./logs/', graph=tf.get_default_graph())
        self.sess.run(tf.global_variables_initializer())

    def gaussian_kernel_function(self, center, data_point):
        return np.exp(-self.beta * np.square(data_point - center))

    def featurizer(self, center, observation):
        features = self.gaussian_kernel_function(center, observation)
        return features

    def object_function(self):
        # define inputs
        with tf.name_scope("inputs"):
            self.tf_obs = tf.placeholder(tf.float32, shape=[None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, shape=[None, ], name="actions")
            self.tf_vl = tf.placeholder(tf.float32, shape=[None, ], name="value")

        # fully connected layer 01
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=10,
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name="fc01"
        )
        # ouput layer
        output_layer = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='output'
        )
        #
        self.acts_prob = tf.nn.softmax(output_layer, name="actions_prob")
        #
        with tf.name_scope("loss"):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.acts_prob, labels=self.tf_acts)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vl)

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def get_action(self, observation):
        prob_weights = self.sess.run(self.acts_prob, feed_dict={self.tf_obs: observation})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def get_paths(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_acts.append(a)
        self.ep_rs.append(float(r))

    def train(self):
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        self.sess.run(self.train_op,
                      feed_dict={
                          self.tf_obs: np.vstack(self.ep_obs),
                          self.tf_acts: np.array(self.ep_acts),
                          self.tf_vl: discounted_ep_rs_norm,
                      })

        self.ep_obs = []
        self.ep_acts = []
        self.ep_rs = []
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount eposide rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

if __name__ == '__main__':
    RENDER = False
    #
    env = gym.make("NChain-v0")
    env.seed(1)
    env = env.unwrapped
    #
    policy = REINFORCE(n_actions=env.action_space.n,
                       n_features=env.n,
                       learning_rate=0.02,
                       reward_decay=0.99,
                       outuput_graph=False)
    for i_episode in range(1000):
        observation = env.reset()
        count = 0
        # sample
        while True:
            count += 1
            # if RENDER:
            #     env.render()
            # print(np.array(observation).reshape(1,-1))
            obs_tmp = np.zeros(shape=env.n)
            obs_tmp[observation] = 1
            obs_feed = obs_tmp.reshape(1, -1)
            # print(obs_feed)
            action = policy.get_action(obs_feed)
            next_observation, reward, done, _ = env.step(action)
            policy.get_paths(obs_feed, action, reward)

            if count == 100:
                ep_rs_rum = sum(policy.ep_rs)

                if 'running_reward' not in globals():
                    running_reward = ep_rs_rum
                else:
                    running_reward = running_reward * 0.99 + ep_rs_rum * 0.01

                print("\repisode : ", i_episode, " reward : ", int(running_reward))
                # if running_reward > -500:
                #     RENDER=True
                value = policy.train()
                break
            observation = next_observation
