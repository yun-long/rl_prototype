import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')
import gym
import os
import ot
import pandas as pd
#
import rl.algo.ppo.constant as C
import rl.tf.baselines.common.tf_util as U
from rl.algo.ppo.utils import build_mlp, alpha_fn, f_div, w2
from rl.sampler.advanced_sampler import AdvancedSampler, next_batch_idx
from rl.tf.baselines.common.misc_util import zipsame

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
        #
        self.sample_action = self.pd.sample()

    def act(self, state):
        action = np.squeeze(self.sess.run(self.sample_action, {self.X: np.asmatrix(state)}), axis=0)
        return action

    def get_params(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_name)


class PPO(object):
    def __init__(self, env, sess, method, **kwargs):
        self.env = env
        self.sess = sess
        self.method = method
        self.act_space = env.action_space
        self.obs_space = env.observation_space
        self.states = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_space.shape[0]], name='states')
        #
        self.__define_actor(method=method, **kwargs)
        self.__define_critic(**kwargs)
        #

    def __define_actor(self, method, a_lrate=1e-3, beta=3.0, clip_param=0.2, ent_coeff=0.0):
        with tf.variable_scope('actor'):
            self.pi = MlpPolicy(sess, self.states, self.act_space,
                                name='mlp_pi', trainable=True) # new policy
            self.oldpi = MlpPolicy(sess, self.states, self.act_space,
                                   name='mlp_oldpi', trainable=False) # old policy
            self.old_logp_act = tf.placeholder(tf.float32, shape=[None, self.act_space.shape[0]], name='old_logp_act')
            self.target_adv = tf.placeholder(tf.float32, shape=[None, 1], name='advantages')
            self.actions = tf.placeholder(tf.float32, shape=[None, self.act_space.shape[0]], name='action')
            self.logp_act = self.pi.pd.log_prob(self.actions)
            self.p_act = self.pi.pd.prob(self.actions)
            ratio = tf.exp(self.logp_act - self.old_logp_act)
            entropy = self.pi.pd.entropy()
            self.meanent = tf.reduce_mean(entropy)
            clip_pr = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param)
            with tf.variable_scope('loss_optimizer'):
                if method == 'clip':
                    self.loss_policy = - tf.reduce_mean(tf.minimum(x=tf.multiply(ratio, self.target_adv),
                                                              y=tf.multiply(clip_pr, self.target_adv)))
                    kl = tf.contrib.distributions.kl_divergence(self.oldpi.pd, self.pi.pd)
                    self.mean_div = tf.reduce_mean(kl)
                elif method == 'kl':
                    kl = tf.contrib.distributions.kl_divergence(self.oldpi.pd, self.pi.pd)
                    self.mean_div = tf.reduce_mean(kl)
                    self.loss_policy = - (tf.reduce_mean(ratio * self.target_adv) - beta * self.mean_div)
                elif method == 'f':
                    f_kl = alpha_fn(alpha=alpha) # alpha=1.0, kl divergence
                    # approximate f divergence
                    f_divergence = f_div(f_kl, log_p=self.old_logp_act, log_q=self.logp_act)
                    self.mean_div = tf.reduce_mean(f_divergence)
                    # TODO: make beta smaller
                    self.loss_policy = -(tf.reduce_mean(ratio * self.target_adv) - beta * self.mean_div)
                elif method.lower() == 'w2':
                    self.w2 = tf.placeholder(tf.float32, shape=1, name='wasserstein2_distance')
                    # mean_w2 = tf.reduce_mean()
                    self.loss_policy = -(tf.reduce_mean(ratio * self.target_adv) - beta * self.w2)
                else:
                    raise NotImplementedError
            self.train_opt_policy = tf.train.AdamOptimizer(a_lrate).minimize(self.loss_policy)
        self.assign_old_eq_new = U.function([], [],
                                            updates=[tf.assign(oldv, newv) for (oldv, newv) in
                                                     zipsame(self.oldpi.get_params(), self.pi.get_params())])


    def __define_critic(self, c_lrate=3e-4):
        with tf.variable_scope('critic'):
            self.valuf_fn = MlpValue(sess, self.states, name='mlp_value', trainable=True)
            self.target_v = tf.placeholder(tf.float32, shape=[None, 1], name='target_v')
            with tf.variable_scope("loss_optimizer"):
                self.loss_value = tf.losses.mean_squared_error(self.target_v, self.valuf_fn.vf)
                self.train_opt_value = tf.train.AdamOptimizer(c_lrate).minimize(self.loss_value)

    def get_log_prob(self, state, action):
        return self.sess.run(self.logp_act, {self.pi.X: state, self.actions: action})

    def get_prob(self, state, action):
        return self.sess.run(self.p_act, {self.pi.X: state, self.actions: action})

    def train_actor(self, state, action, advantage, old_log_prob, w2=None):
        if w2 is not None:
            feed_dict = {self.pi.X: state,
                         self.actions: action,
                         self.target_adv: advantage,
                         self.old_logp_act: old_log_prob,
                         self.w2: w2}
            return self.sess.run([self.loss_policy, self.w2, self.meanent, self.train_opt_policy],
                            feed_dict=feed_dict)
        else:
            feed_dict = {self.pi.X: state,
                         self.actions: action,
                         self.target_adv: advantage,
                         self.old_logp_act: old_log_prob}
            return self.sess.run([self.loss_policy, self.mean_div, self.meanent, self.train_opt_policy],
                            feed_dict=feed_dict)

    def train_critic(self, state, value):
        feed_dict = {self.pi.X:state, self.target_v: value}
        return self.sess.run([self.loss_value, self.train_opt_value],
                             feed_dict=feed_dict)

def learn(env, sess, i_trial, num_episodes, max_trans,
          epochs, batch_size, gamma, lam, kl_target, w2_target, method, alpha, **kwargs):
    ppo = PPO(env=env, sess=sess, method=method, **kwargs)
    if method.lower() == 'w2':
        beta = 0.1
        div_target = w2_target
    else:
        beta = 3.0
        div_target = kl_target
    #
    sampler = AdvancedSampler(env)
    sess.run(tf.global_variables_initializer())
    for i_episode in range(num_episodes):
        print("--------method: {}, \t alpha:{}, \t trail: {}, \t episode: {}------------".format(method, alpha,i_trial, i_episode))
        # sample data
        paths = sampler.rollous(ppo.pi.act, n_trans=max_trans)
        mean_r = np.sum(paths['reward']/paths['n_path'])
        print("Mean reward: {}".format(np.sum(paths['reward'] / paths['n_path'])))
        # update critic
        losses_c = []
        for epoch in range(epochs):
            # generate advantages
            adv, values = sampler.get_adv(paths, ppo.valuf_fn.value, discount=gamma, lam=lam)
            v_targes = values + adv
            for batch_idx in next_batch_idx(batch_size, len(v_targes)):
                loss_v, _ = ppo.train_critic(state=paths['state'][batch_idx], value=v_targes[batch_idx])
                losses_c.append(loss_v)
        # writer.add_summary(np.mean(losses), i_episode)
        print("critic: ",np.mean(losses_c))

        # update actor
        adv, _ = sampler.get_adv(paths, ppo.valuf_fn.value, discount=gamma, lam=lam)
        adv = adv / np.std(adv)
        #
        logp_act_sample = ppo.get_log_prob(state=paths['state'], action=paths['action'])
        p_act_smaple = ppo.get_prob(state=paths['state'], action=paths['action'])
        ppo.assign_old_eq_new()
        #
        losses_a = []
        divg_tmp = []
        ent_tmp = []
        for epoch in range(epochs):
            for batch_idx in next_batch_idx(batch_size, len(adv)):
                if method == 'w2':
                    action = paths['action'][batch_idx]
                    p = p_act_smaple[batch_idx]
                    q = ppo.get_prob(state=paths['state'], action=paths['action'])[batch_idx]
                    dist_w = w2(action=action, p=p, q=q)
                else:
                    dist_w = None
                loss_a, d, ent, _ = ppo.train_actor(state=paths['state'][batch_idx],
                                                    action=paths['action'][batch_idx],
                                                    advantage=adv[batch_idx],
                                                    old_log_prob=logp_act_sample[batch_idx],
                                                    w2=dist_w)
                losses_a.append(loss_a)
                divg_tmp.append(d)
                ent_tmp.append(ent)
                if method == 'kl' and d > 4 * div_target:
                    print("breaking")
                    break
            if d < div_target / 1.5:
                beta /= 2
            elif d > div_target * 1.5:
                beta *= 2
            beta = np.clip(beta, 1e-4, 10)
        print("actor: ", np.mean(losses_a))
        data_series = pd.Series([method, alpha, i_trial, i_episode, mean_r, np.mean(losses_c), np.mean(losses_a), np.mean(divg_tmp), np.mean(ent_tmp), beta], index=C.columns)
        C.data = C.data.append(data_series, ignore_index=True)
    # save data after each iteration
    C.data.to_csv(C.path_csv, index=False)


if __name__ == '__main__':
    env = gym.make(C.env_ID)
    #
    for method, values in C.params['methods'].items():
        for i, alpha in enumerate(values):
            if alpha == None:
                alpha = method
            mean_rewards = np.zeros(shape=(C.params['num_trials'], C.params['num_episodes']))
            env.seed(C.seed)
            tf.set_random_seed(C.seed)
            np.random.seed(C.seed)
            for i_trial in range(C.params['num_trials']):
                tf.reset_default_graph()
                with tf.Session() as sess:
                    learn(env=env,
                          sess=sess,
                          i_trial=i_trial,
                          num_episodes=C.params['num_episodes'],
                          max_trans=C.params['num_sample_trans'],
                          epochs=C.params['epochs'],
                          batch_size=C.params['batch_size'],
                          gamma=C.params['gamma'],
                          lam=C.params['lam'],
                          kl_target=C.params['kl_target'],
                          w2_target=C.params['w2_target'],
                          method=method,
                          alpha=alpha)



