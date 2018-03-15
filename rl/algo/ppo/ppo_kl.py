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
from rl.misc.utilies import get_dirs
from rl.algo.ppo.utils import build_mlp, alpha_fn, f_div, ppo_result_path, w2
from rl.sampler.advanced_sampler import AdvancedSampler, next_batch_idx
from rl.tf.baselines.common.misc_util import zipsame
import rl.tf.baselines.common.tf_util as U

columns = ['methods', 'alphas', 'trials', 'episodes', 'rewards', 'losses_c', 'losses_a', 'divergences', 'entropies']
data = pd.DataFrame(columns= columns)

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

def learn(env, sess, i_trial,
          num_episodes=100,
          max_trans=3200,
          epochs=10,
          batch_size=32,
          gamma=0.99,
          lam=0.95,
          clip_param=0.2,
          beta=3.0,
          ent_coeff=0.0,
          c_lrate=3e-4,
          a_lrate=3e-4,
          kl_target=0.01,
          w2_target=0.1,
          alpha=1.0, method='kl'):
    assert isinstance(method, str)
    if method.lower() == 'w2':
        div_target = w2_target
        beta = 0.1
    else:
        beta = 3.0
        div_target = kl_target
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
        old_p_act = tf.placeholder(tf.float32, shape=[None, 1], name='old_p_act')
        target_adv = tf.placeholder(tf.float32, shape=[None, 1], name='advantages')
        samp_action = tf.placeholder(tf.float32, shape=[None, act_space.shape[0]], name='action')
        # log_prob = tf.expand_dims(pi.pd.log_prob(samp_action), axis=-1)
        log_prob = pi.pd.log_prob(samp_action)
        prob = pi.pd.prob(samp_action)
        #
        ratio = tf.exp(log_prob - old_logp_act)
        entropy = pi.pd.entropy()
        meanent = tf.reduce_mean(entropy)
        clip_pr = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param)
        with tf.variable_scope("loss_optimizer"):
            if method == 'clip':
                loss_policy = - tf.reduce_mean(tf.minimum(x=tf.multiply(ratio, target_adv),
                                                          y=tf.multiply(clip_pr, target_adv)))
                kl = tf.contrib.distributions.kl_divergence(oldpi.pd, pi.pd)
                mean_div = tf.reduce_mean(kl)
            elif method == 'kl':
                kl = tf.contrib.distributions.kl_divergence(oldpi.pd, pi.pd)
                mean_div = tf.reduce_mean(kl)
                loss_policy = - (tf.reduce_mean(ratio * target_adv) - beta * mean_div)
            elif method == 'f':
                f_kl = alpha_fn(alpha=alpha) # alpha=1.0, kl divergence
                # approximate f divergence
                f_divergence = f_div(f_kl, log_p=old_logp_act, log_q=log_prob)
                mean_div = tf.reduce_mean(f_divergence)
                # TODO: make beta smaller
                loss_policy = -(tf.reduce_mean(ratio * target_adv) - beta * mean_div)
            elif method.lower() == 'w2':
                ws2 = tf.placeholder(tf.float32, shape=1, name='wasserstein2_distance')
                # mean_w2 = tf.reduce_mean()
                loss_policy = -(tf.reduce_mean(ratio * target_adv) - beta * ws2)
            else:
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

    def get_prob(state, action):
        return sess.run(prob, {pi.X: state, samp_action: action})

    def train_actor(state, action, advantage, old_log_prob, sample_ws2=None):
        if sample_ws2 is not None:
            feed_dict = {pi.X: state, samp_action: action, target_adv: advantage, old_logp_act: old_log_prob, ws2: sample_ws2}
            return sess.run([loss_policy, ws2, meanent, train_opt_policy], feed_dict=feed_dict)
        else:
            feed_dict = {pi.X: state, samp_action: action, target_adv: advantage, old_logp_act: old_log_prob}
            return sess.run([loss_policy, mean_div, meanent, train_opt_policy], feed_dict=feed_dict)

    def train_critic(state, value):
        feed_dict = {pi.X: state, target_v: value}
        return sess.run([loss_value, train_opt_value], feed_dict=feed_dict)


    sampler = AdvancedSampler(env)
    sess.run(tf.global_variables_initializer())
    # saver =  tf.train.Saver()
    # writer = tf.summary.FileWriter(log_path, sess.graph)
    for i_episode in range(num_episodes):
        print("--------method: {}, \t alpha:{}, \t trail: {}, \t episode: {}------------".format(method, alpha,i_trial, i_episode))
        # sample data
        paths = sampler.rollous(pi.act, n_trans=max_trans)
        mean_r = np.sum(paths['reward']/paths['n_path'])
        print("Mean reward: {}".format(np.sum(paths['reward'] / paths['n_path'])))
        # update critic
        losses_c = []
        for epoch in range(epochs):
            # generate advantages
            adv, values = sampler.get_adv(paths, valuf_fn.value, discount=gamma, lam=lam)
            v_targes = values + adv
            for batch_idx in next_batch_idx(batch_size, len(v_targes)):
                loss_v, _ = train_critic(state=paths['state'][batch_idx], value=v_targes[batch_idx])
                losses_c.append(loss_v)
        # writer.add_summary(np.mean(losses), i_episode)
        print("critic: ",np.mean(losses_c))

        # update actor
        adv, _ = sampler.get_adv(paths, valuf_fn.value, discount=gamma, lam=lam)
        adv = adv / np.std(adv)
        #
        logp_act_sample = get_log_prob(state=paths['state'], action=paths['action'])
        p_act_smaple = get_prob(state=paths['state'], action=paths['action'])
        assign_old_eq_new()
        #
        losses_a = []
        divg_tmp = []
        ent_tmp = []
        for epoch in range(epochs):
            for batch_idx in next_batch_idx(batch_size, len(adv)):
                if method == 'w2':
                    action = paths['action'][batch_idx]
                    p = p_act_smaple[batch_idx]
                    q = get_prob(state=paths['state'], action=paths['action'])[batch_idx]
                    dist_w = w2(action=action, p=p, q=q)
                else:
                    dist_w = None
                loss_a, d, ent, _ = train_actor(state=paths['state'][batch_idx],
                                              action=paths['action'][batch_idx],
                                              advantage=adv[batch_idx],
                                              old_log_prob=logp_act_sample[batch_idx],
                                              sample_ws2=dist_w)
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
        # writer.add_summary(np.mean(losses), i_episode)
        print("actor: ", np.mean(losses_a))
        if alpha == 1.0:
            tmp_alpha = 'KL'
        else:
            tmp_alpha = alpha
        data_series = pd.Series([method, tmp_alpha, i_trial, i_episode, mean_r, np.mean(losses_c), np.mean(losses_a), np.mean(divg_tmp), np.mean(ent_tmp)], index=columns)
        global data
        data = data.append(data_series, ignore_index=True)


def run_trails(env, num_trails, num_episodes, method, alpha, seed):
    mean_rewards = np.zeros(shape=(num_trails, num_episodes))
    env.seed(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    for i_trial in range(num_trails):
        tf.reset_default_graph()
        with tf.Session() as sess:
            learn(env=env, sess=sess, i_trial=i_trial, num_episodes=num_episodes, alpha=alpha, method=method)
    return mean_rewards

if __name__ == '__main__':
    seed = 1234
    # env_ID = "Pendulum-v0"
    env_ID = "MountainCarContinuous-v0"
    path_result = get_dirs(os.path.join(ppo_result_path, 'pendulum'))
    path_csv = os.path.join(path_result, 'data.csv')
    env = gym.make(env_ID)
    #
    methods = ['f', 'clip']
    # methods = ['w2']
    alphas = ['GAN', 1.0, 2.0]
    num_trails = 5
    num_episodes = 100
    for i_m, method in enumerate(methods):
        if method == 'clip' or method == 'w2':
            if method == 'w2':
                num_trails = 2
            stats = run_trails(env, num_trails, num_episodes, method=method, alpha=method, seed=seed)
            data.to_csv(path_csv, index=False)
        elif method == 'f':
            for i_alpha, alpha in enumerate(alphas):
                stats = run_trails(env, num_trails, num_episodes, method=method, alpha=alpha, seed=seed)
                #
                data.to_csv(path_csv, index=False)



