import tensorflow as tf
import numpy as np
import gym
import rl.tf.baseline.tf_util as U
from rl.tf.baseline.tf_util import function, get_placeholder_cached, flatgrad
from rl.tf.baseline.distributions import make_pdtype
from rl.tf.baseline.mpi_adam import MpiAdam
from rl.tf.baseline.misc_util import zipsame
from rl.tf.baseline.console_util import fmt_row
from rl.tf.baseline.sampler import traj_segment_generator, add_vtarg_and_adv
from rl.tf.baseline.dataset import Dataset
# from rl.sampler.advanced_sampler import AdvancedSampler

class MlpPolicy(object):

    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, act_space, obs_space, hid_sizes, num_hid_layers, sess=None):
        a_dim = act_space.shape[0]
        s_dim = obs_space.shape[0]
        self.pdtype = pdtype = make_pdtype(act_space)
        sequence = None
        #
        self.state = U.get_placeholder(name='state', dtype=tf.float32, shape=[sequence] + list(obs_space.shape))
        #
        with tf.variable_scope('actor'):
            lastout = self.__mlp(self.state, hid_sizes, num_hid_layers)
            mean = tf.layers.dense(inputs=lastout, units=a_dim, name='mu', kernel_initializer=tf.glorot_uniform_initializer())
            logstd = tf.get_variable(name='logstd', shape=[1, a_dim], initializer=tf.zeros_initializer())
            pdparam = tf.concat([mean, mean * 0. + logstd], axis=1)

        self.pd = pdtype.pdfromflat(pdparam)

        with tf.variable_scope('critic'):
            lastout = self.__mlp(self.state, hid_sizes, num_hid_layers)
            self.vpred = tf.layers.dense(inputs=lastout, units=1, name='value', kernel_initializer=tf.glorot_uniform_initializer())[:, 0]

        self.state_in = []
        self.state_out = []
        self.sess = sess
        act = self.pd.sample()
        self._act = U.function(inputs=[self.state], outputs=act)
        self._v = function(inputs=[self.state], outputs=self.vpred)

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

    def act(self, state):
        state_act = self._act(state[None])
        return state_act[0]

    def value(self, state):
        state_v = self._v(state[None])
        return state_v[0]

    def get_variables(self):
        return tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

    def get_initial_state(self):
        return []


def learn(env, policy_fn, sess, timesteps_per_actorbatch, batch_size, clip_param, entcoeff, gamma, lam,
          n_episode, n_epochs, adam_epsilon=1e-5, optim_stepsize=1e-3):
    #
    act_space = env.action_space
    obs_space = env.observation_space
    #
    pi = policy_fn('pi', act_space, obs_space, sess)
    oldpi = policy_fn('old', act_space, obs_space, sess)
    # some placeholder
    obs = get_placeholder_cached(name='state')
    act = pi.pdtype.sample_placeholder([None])
    # act = tf.placeholder(tf.float32, shape=[None, act_space.shape[0]], name='test_action')
    target_adv = tf.placeholder(tf.float32, shape=[None], name='target_adv')
    target_v = tf.placeholder(tf.float32, shape=[None], name='target_v')
    #
    kloldnew = oldpi.pd.kl(other=pi.pd) # KL divergence between old policy and new policy
    ent = pi.pd.entropy() # entropy of new policy
    meankl = tf.reduce_mean(kloldnew) # mean kl
    meanent = tf.reduce_mean(ent) # mean entropy
    pol_entpen = (-entcoeff) * meanent # policy entropy penalty
    #
    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[])
    clip_param = clip_param * lrmult
    # losses
    ratio = tf.exp(pi.pd.logp(act) - oldpi.pd.logp(act))
    surr1 = ratio * target_adv
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * target_adv
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - target_v))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ['pol_surr', 'pol_entpen', 'vf_loss', 'kl', 'entropy']
    #
    var_list = pi.get_trainable_variables()
    lossandgrad = function(inputs=[obs, act, target_adv, target_v, lrmult],
                           outputs=losses + [flatgrad(total_loss, var_list)])
    #
    adam = MpiAdam(var_list, epsilon=adam_epsilon)
    assign_old_eq_new = function(inputs=[], outputs=[], updates=[tf.assign(oldv, newv)
                                for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_loss = function([obs, act, target_adv, target_v, lrmult], losses)
    #
    sess.run(tf.global_variables_initializer())
    adam.sync()

    writer = tf.summary.FileWriter('log', sess.graph)
    # sampler = AdvancedSampler(env)
    #
    tran_gen = traj_segment_generator(pi=pi, env=env, horizon=timesteps_per_actorbatch)
    for i_episode in range(n_episode):
        print("------------- episode: {} ---------------".format(i_episode))
        # sample data
        cur_lrmult = 1.0
        paths = tran_gen.__next__()
        add_vtarg_and_adv(paths, gamma, lam)
        #
        ob, ac, advantages, tdlamret = paths['ob'], paths['ac'], paths['adv'], paths['tdlamret']
        vpredbefore = paths['vpred']
        #
        advantages = (advantages - advantages.mean()) / advantages.std()
        dataset = Dataset(dict(ob=ob, ac=ac, atarg=advantages, vtarg=tdlamret), shuffle=True)
        #
        assign_old_eq_new()
        # a bunch of optimization
        for i in range(n_epochs):
            losses = []
            for batch in dataset.iterate_once(batch_size):
                *newlosses, g = lossandgrad(batch['ob'], batch['ac'], batch['atarg'], batch['vtarg'], cur_lrmult)
                adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(newlosses)
        print(fmt_row(13, loss_names))
        print(fmt_row(13, np.mean(losses, axis=0)))

        # Evaluating losses...
        losses = []
        for batch in dataset.iterate_once(batch_size):
            newlosses = compute_loss(batch['ob'], batch['ac'], batch['atarg'], batch['vtarg'], cur_lrmult)
            losses.append(newlosses)
        # print(fmt_row(13, np.mean(losses, axis=0)))

def policy_fn(name, act_space, obs_space, sess=None):
    return MlpPolicy(name=name, act_space=act_space, obs_space=obs_space,
                     hid_sizes=64, num_hid_layers=2, sess=sess)

if __name__ == '__main__':
    # env = gym.make("MountainCarContinuous-v0")
    env = gym.make("Pendulum-v0")
    env.seed(1234)
    #
    with tf.Session() as sess:
        learn(env,
              policy_fn,
              sess,
              timesteps_per_actorbatch=2048,
              batch_size = 64,
              clip_param=0.2,
              entcoeff=0.0,
              gamma=0.99,
              lam=0.95,
              n_episode=200,
              n_epochs = 10)

