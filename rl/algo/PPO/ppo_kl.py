from rl.env.env_list import env_IDs
from rl.sampler.advanced_sampler import AdvancedSampler
from rl.env.random_jump import RandomJumpEnv
from rl.misc.utilies import opt_policy_demo
#
import gym
import tensorflow as tf
import numpy as np

# choose PPO version
obj = ['clipping', 'kl_penalty']
#
opts = {'obj_ID': obj[1], # choose the surrogate objectives, 0 = clipping, 1 = kl_penalty
        'n_episod': 50,
        'n_trans': 3000,
        'n_epochs': 15,
        'sigma': 1,
        'discount': 0.99,
        'lam_trace': 0.95,
        'epsilon': 0.2,
        'beta': 5,
        'kl_target': 1,
        'bathch_size': 32,
        'logdir':"/Users/yunlong/Gitlab/rl_prototype/results/ppo/log"}
#
class PPO(object):
    """
    Proximal Policy Optimization
    """

    def __init__(self, env, epsilon=0.2, c_lrate=5e-4, a_lrate=5e-4):
        self.s_dims = env.observation_space.shape[0]
        self.a_dims = env.action_space.shape[0]
        #
        self.sess = tf.Session()
        self.states = tf.placeholder(tf.float32, shape=[None, self.s_dims], name="states")

        # actor and critic
        self.__define_critic(c_lrate=c_lrate)
        self.__define_actor(a_lrate=a_lrate)

        self.writer = tf.summary.FileWriter(opts['logdir'], self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def __define_critic(self, c_lrate):
        # define value function, multiplelayers perceptron
        val_nn_layers = [64, 64, 1]
        val_nn_activ  = [tf.nn.relu, tf.nn.tanh, tf.identity]
        with tf.variable_scope("critic"):
            self.target_v = tf.placeholder(tf.float32, shape=[None, 1], name='target_v')
            self.v = self.__build_mlp(self.states, val_nn_layers, val_nn_activ, name='value', trainable=True)
            with tf.variable_scope("loss"):
                self.loss_c = tf.losses.mean_squared_error( self.target_v, self.v)
            with tf.variable_scope("optimizer"):
                self.train_opt_c = tf.train.AdamOptimizer(c_lrate).minimize(self.loss_c)

    def __define_actor(self, a_lrate):
        # define policy function, multiplelayers perceptron
        pol_nn_layers = [64, 64, self.a_dims]
        pol_nn_activ  = [tf.nn.relu, tf.nn.relu, tf.nn.tanh]
        #
        with tf.variable_scope("actor"):
            self.old_log_probas = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='old_log_prob')
            self.actions = tf.placeholder(tf.float32, shape=[None, self.a_dims], name="actions")
            self.target_adv = tf.placeholder(tf.float32, shape=[None, 1], name='target_adv')
            self.sigma = tf.exp(tf.Variable(np.log(opts['sigma']) * tf.ones([1, self.a_dims])))
            pi, pi_params = self.__build_mlp(self.states, pol_nn_layers, pol_nn_activ, name='pi', trainable=True)
            old_pi, old_pi_params = self.__build_mlp(self.states, pol_nn_layers, pol_nn_activ, name='old_pi', trainable=False)
            #
            self.log_prob = tf.expand_dims(pi.log_prob(self.actions), axis=-1)
            # old_log_prob = np.expand_dims(old_pi.log_prob(self.actions), axis=-1)
            with tf.variable_scope("loss"):
                # ratio = pi.prob(self.actions) / old_pi.prob(self.actions)
                ratio = tf.exp(self.log_prob - self.old_log_probas)
                if opts['obj_ID'] == 'kl_penalty':
                    # ratio = pi.prob(self.actions) / old_pi.prob(self.actions)
                    kl = tf.contrib.distributions.kl_divergence(pi, old_pi)
                    self.kl_mean = tf.reduce_mean(kl)
                    self.loss_a = - tf.reduce_mean(ratio * self.target_adv) -  opts['beta'] * kl
                elif opts['obj_ID'] == 'clipping':
                    self.clip_pr = tf.clip_by_value(ratio, 1.-opts['epsilon'], 1.+opts['epsilon'])
                    self.loss_a = - tf.reduce_mean(tf.minimum(x=tf.multiply(ratio, self.target_adv),
                                                              y=tf.multiply(self.clip_pr, self.target_adv)))
            #
            with tf.variable_scope("optimizer"):
                self.train_opt_a = tf.train.AdamOptimizer(a_lrate).minimize(self.loss_a)
            #
            with tf.variable_scope("update_old_pi"):
                self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, old_pi_params)]
            #
            with tf.variable_scope("sample_action"):
                self.sample_action =  pi.sample()

    def __build_mlp(self, input, sizes, activations, name, trainable):
        """
        Build multilayer perceptron
        """
        with tf.variable_scope(name):
            last_out = input
            for l, size in enumerate(sizes):
                last_out = tf.layers.dense(inputs=last_out,
                                           units=size,
                                           activation=activations[l],
                                           kernel_initializer=tf.glorot_uniform_initializer(),
                                           trainable=trainable)
            out = last_out
            params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
            if name == "value":
                return out
            else:
                norm_dist = tf.contrib.distributions.MultivariateNormalDiag(out, self.sigma)
                return norm_dist, params

    def get_log_prob(self, states, actions):
        feed_dict = {self.states: states,
                     self.actions: actions}
        return self.sess.run(self.log_prob, feed_dict)

    def predict_action(self, states):
        action = np.squeeze(self.sess.run(self.sample_action, {self.states: np.asmatrix(states)}), axis=0)
        # print(action)
        return action

    def predict_value(self, states):
        value = self.sess.run(self.v, {self.states: states})
        # print(value)
        return value

    def update_oldpi(self):
        self.sess.run(self.update_oldpi_op)

    def train_critic(self, states, target_v):
        feed_dict = {self.states: states,
                     self.target_v: target_v}
        return self.sess.run(self.train_opt_c, feed_dict)

    def train_actor(self, states, actions, target_adv, old_log_prob):
        feed_dict = {self.states: states,
                     self.actions: actions,
                     self.old_log_probas: old_log_prob,
                     self.target_adv: target_adv}
        if opts['obj_ID'] == 'kl_penalty':
            return self.sess.run([self.train_opt_a, self.kl_mean], feed_dict)
        else:
            return self.sess.run(self.train_opt_a, feed_dict), 10

    def eval_critic(self, states, target_v):
        feed_dict = {self.states: states,
                     self.target_v: target_v}
        return self.sess.run(self.loss_c, feed_dict)

    def eval_actor(self, states, action):
        feed_dict = {self.states: states,
                     self.actions: action}
        return self.sess.run(self.loss_a, feed_dict)

def next_batch_idx(batch_size, data_set_size):
    batch_idx_list = np.random.choice(data_set_size, data_set_size, replace=False)
    for batch_start in range(0, data_set_size, batch_size):
        yield batch_idx_list[batch_start: min(batch_start+batch_size, data_set_size)]


def main():
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)
    #
    # env = gym.make(env_IDs.Pendulum)
    env = RandomJumpEnv()
    env.seed(seed)
    # define Proximal policy optimization class
    ppo = PPO(env=env)
    # define sampler
    sampler = AdvancedSampler(env)
    # initilization
    #
    for i_episode in range(opts['n_episod']):
        print("-------- episode: {}------------".format(i_episode))
        # sample data
        paths = sampler.rollous(ppo.predict_action, n_trans=opts['n_trans'])
        print("Mean reward: {}".format(np.mean(paths['reward'])))

        # update critic
        #
        for epoch in range(opts['n_epochs']):
            # generate advantages
            adv, values = sampler.get_adv(paths, ppo.predict_value, discount=opts['discount'], lam=opts['lam_trace'])
            v_targes = values + adv
            #
            if epoch == 0:
                print("Loss value function: {}".format(ppo.eval_critic(paths['state'], v_targes)))
            for batch_idx in next_batch_idx(opts['bathch_size'], len(v_targes)):
                ppo.train_critic(paths['state'][batch_idx], v_targes[batch_idx])

        # update actor
        adv, _ = sampler.get_adv(paths, ppo.predict_value, discount=opts['discount'], lam=opts['lam_trace'])
        adv = adv / np.std(adv)
        #
        log_act_prob = ppo.get_log_prob(paths['state'], paths['action'])
        ppo.update_oldpi()
        #
        for epoch in range(opts['n_epochs']):
            for batch_idx in next_batch_idx(opts['bathch_size'], len(adv)):
                _, d_kl = ppo.train_actor(paths['state'][batch_idx],
                                          paths['action'][batch_idx],
                                          adv[batch_idx],
                                          old_log_prob=log_act_prob[batch_idx])
            # if d_kl > 4 * opts['kl_target']:
            #     break

        # if opts['obj_ID'] == 'kl_penalty':
        #     if d_kl < opts['kl_target'] / 1.5:
        #         opts['beta'] /=2
        #     elif d_kl > opts['kl_target'] * 1.5:
        #         opts['beta'] *= 2
    #
    opt_policy_demo(env, policy=ppo.predict_action)
    ppo.writer.close()

if __name__ == '__main__':
    main()