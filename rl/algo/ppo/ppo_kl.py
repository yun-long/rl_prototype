import tensorflow as tf
import numpy as np
import gym
import os
#
from rl.misc.utilies import get_dirs
from rl.misc.plot_rewards import plot_tr_ep_rs, plot_coeff_tr_ep_rs
from rl.misc.utilies import opt_policy_demo
from rl.sampler.advanced_sampler import AdvancedSampler, next_batch_idx
from rl.tf.baselines.common.misc_util import zipsame
import rl.tf.baselines.common.tf_util as U
#
result_path = os.path.join(os.path.realpath("../../../"), 'results')
ppo_result_path = get_dirs(os.path.join(result_path, 'ppo'))
model_path = get_dirs(os.path.join(ppo_result_path, 'model'))
log_path = get_dirs(os.path.join(ppo_result_path, 'log'))

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

class MlpValue(object):
    def __init__(self, sess, input, name, trainable=True):
        self.sess = sess
        self.X = input
        sizes = [64, 64, 1]
        activations = [tf.nn.relu, tf.nn.relu, tf.identity]
        with tf.variable_scope(name):
            self.vf = build_mlp(input, sizes, activations, trainable=trainable)

    def value(self, state):
        return self.sess.run(self.vf, {self.X: state})

class MlpPolicy(object):
    def __init__(self, sess, input, ac_space, name, trainable=True):
        self.sess = sess
        self.X = input
        actdim = ac_space.shape[0]
        sizes = [64, 64, actdim]
        activations = [tf.nn.relu, tf.nn.relu, tf.nn.tanh]
        with tf.variable_scope(name):
            mean = build_mlp(input, sizes, activations, trainable=trainable)
            std = tf.exp(tf.get_variable(name="logstd", shape=[1, actdim], initializer=tf.ones_initializer()))
            # self.pd = tf.contrib.distributions.MultivariateNormalDiag(mean, std)
            self.pd = tf.distributions.Normal(loc=mean, scale=std)
            self.scope_name = tf.get_variable_scope().name
            # print(self.scope_name)
        #
        self.sample_action = self.pd.sample()

    def act(self, state):
        action = np.squeeze(self.sess.run(self.sample_action, {self.X: np.asmatrix(state)}), axis=0)
        return action

    def get_params(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_name)


def policy_fn(sess, input, ac_space, name, trainable):
    mlp =  MlpPolicy(sess=sess, input=input, ac_space= ac_space, name=name, trainable=trainable)
    params = mlp.get_params()
    return mlp, params

def learn(env, sess, episodes, max_trans, epochs, batch_size, gamma, lam, clip_param, beta, ent_coeff, c_lrate, a_lrate, kl_target, method='kl'):
    act_space = env.action_space
    obs_space = env.observation_space
    #
    samp_state = tf.placeholder(dtype=tf.float32, shape=[None, obs_space.shape[0]], name='states')
    #
    with tf.variable_scope("actor"):
        pi = MlpPolicy(sess, samp_state, act_space, name='mlp_pi', trainable=True) # new policy
        oldpi = MlpPolicy(sess, samp_state, act_space, name='mlp_oldpi', trainable=False) # old policy
        #
        old_logp_act = tf.placeholder(tf.float32, shape=[None, 1], name='old_logp_act')
        target_adv = tf.placeholder(tf.float32, shape=[None, 1], name='advantages')
        samp_action = tf.placeholder(tf.float32, shape=[None, act_space.shape[0]], name='action')
        # log_prob = tf.expand_dims(pi.pd.log_prob(samp_action), axis=-1)
        log_prob = pi.pd.log_prob(samp_action)
        #
        ratio = tf.exp(log_prob - old_logp_act)
        entropy = pi.pd.entropy()
        kl = tf.contrib.distributions.kl_divergence(oldpi.pd, pi.pd)
        meankl = tf.reduce_mean(kl)
        meanent = tf.reduce_mean(entropy)
        clip_pr = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param)
        with tf.variable_scope("loss_optimizer"):
            if method == 'clip':
                loss_policy = - tf.reduce_mean(tf.minimum(x=tf.multiply(ratio, target_adv),
                                                          y=tf.multiply(clip_pr, target_adv)))
            elif method == 'kl':
                loss_policy = - (tf.reduce_mean(ratio * target_adv) - beta * meankl)
            else:
                loss_policy = None
                raise NotImplementedError
            train_opt_policy = tf.train.AdamOptimizer(a_lrate).minimize(loss_policy)

    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv) for (oldv, newv) in zipsame(oldpi.get_params(), pi.get_params())])
    with tf.variable_scope("critic"):
        valuf_fn = MlpValue(sess, samp_state, name='mlp_value', trainable=True)
        target_v = tf.placeholder(tf.float32, shape=[None, 1], name='target_v')
        with tf.variable_scope("loss_optimizer"):
            loss_value = tf.losses.mean_squared_error(target_v, valuf_fn.vf)
            train_opt_value = tf.train.AdamOptimizer(c_lrate).minimize(loss_value)


    def get_log_prob(state, action):
        return sess.run(log_prob, {pi.X: state, samp_action: action})

    def train_actor(state, action, advantage, old_log_prob):
        feed_dict = {pi.X: state, samp_action: action, target_adv: advantage, old_logp_act: old_log_prob}
        return sess.run([loss_policy, meankl, train_opt_policy], feed_dict=feed_dict)

    def train_critic(state, value):
        feed_dict = {pi.X: state, target_v: value}
        return sess.run([loss_value, train_opt_value], feed_dict=feed_dict)


    sampler = AdvancedSampler(env)
    sess.run(tf.global_variables_initializer())
    saver =  tf.train.Saver()
    writer = tf.summary.FileWriter(log_path, sess.graph)
    rewards = []
    for i_episode in range(episodes):
        print("-------- episode: {}------------".format(i_episode))
        # sample data
        paths = sampler.rollous(pi.act, n_trans=max_trans)
        print("Mean reward: {}".format(np.sum(paths['reward'] / paths['n_path'])))
        rewards.append(np.sum(paths['reward'] / paths['n_path']))
        #
        losses = []
        for epoch in range(epochs):
            # generate advantages
            adv, values = sampler.get_adv(paths, valuf_fn.value, discount=gamma, lam=lam)
            v_targes = values + adv
            for batch_idx in next_batch_idx(batch_size, len(v_targes)):
                loss_v, _ = train_critic(state=paths['state'][batch_idx], value=v_targes[batch_idx])
                losses.append(loss_v)
        print("critic: ",np.mean(losses))

        # update actor
        adv, _ = sampler.get_adv(paths, valuf_fn.value, discount=gamma, lam=lam)
        adv = adv / np.std(adv)
        #
        logp_act_sample = get_log_prob(state=paths['state'], action=paths['action'])
        assign_old_eq_new()
        #
        losses = []
        for epoch in range(epochs):
            for batch_idx in next_batch_idx(batch_size, len(adv)):
                loss_a, d_kl, _ = train_actor(state=paths['state'][batch_idx],
                                              action=paths['action'][batch_idx],
                                              advantage=adv[batch_idx],
                                              old_log_prob=logp_act_sample[batch_idx])
                losses.append(loss_a)
                if method == 'kl' and d_kl > 4 * kl_target:
                    print("breaking")
                    break
            if d_kl < kl_target / 1.5:
                beta /= 2
            elif d_kl > kl_target * 1.5:
                beta *= 2
            beta = np.clip(beta, 1e-4, 10)
        print("actor: ", np.mean(losses))

    return rewards, saver

if __name__ == '__main__':
    seed = 1234
    env = gym.make("Pendulum-v0")
    pendulum_model = get_dirs(os.path.join(model_path, 'Pendulum-v0'))
    # env = gym.make("MountainCarContinuous-v0")
    #
    methods = ['kl', 'clip']
    num_trails = 10
    num_episodes = 100
    mean_rewards = np.zeros((num_episodes, num_trails, len(methods)))
    for i_m, method in enumerate(methods):
        for i_trail in range(num_trails):
            env.seed(seed)
            tf.reset_default_graph()
            with tf.Session() as sess:
                reward, saver = learn(env, sess,
                                      episodes=num_episodes,
                                      max_trans=3200,
                                      epochs=10,
                                      batch_size=32,
                                      gamma=0.99,
                                      lam=0.95,
                                      clip_param=0.2,
                                      beta=1.0,
                                      ent_coeff=0.0,
                                      c_lrate=3e-4,
                                      a_lrate=3e-4,
                                      kl_target=0.01,
                                      method=method)
                mean_rewards[:, i_trail, i_m] = reward
                if i_trail >= num_trails - 1:
                    if method == 'clip':
                        saver.save(sess, pendulum_model + "/clip_model")
                    if method == 'kl':
                        saver.save(sess, pendulum_model + "/kl_model")
                    sess.close()
    # plot_tr_ep_rs(mean_rewards, show=True)
    plot_coeff_tr_ep_rs(mean_rewards, methods, label='ppo_', savepath=ppo_result_path + '/result.png')


