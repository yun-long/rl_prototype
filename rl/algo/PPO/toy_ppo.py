from rl.policy.gp_mlp import GaussianPolicy_MLP
from rl.tf.models import MLP
from rl.env.random_jump import RandomJumpEnv
#
import gym
import tensorflow as tf
import numpy as np

class PPO(object):
    """
    Proximal Policy Optimization
    """
    def __init__(self, sess, policy, vfunc, epsilon=0.2, a_lrate=5e-4, v_lrate=5e-4):
        self.sess = sess
        self.pol = policy
        self.v = vfunc

        # loss for v function
        self.target_v = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='target_v')
        self.loss_v = tf.losses.mean_squared_error(self.v.out, self.target_v)
        self.optimizer_v = tf.train.AdamOptimizer(v_lrate).minimize(self.loss_v)

        # clip loss for policy update
        self.advantage = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='advantage')
        self.old_log_probas = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='old_log_probas')
        proba_ratio = tf.exp(self.pol.log_prob - self.old_log_probas)
        self.clip_pr = tf.clip_by_value(proba_ratio, 1. - epsilon, 1 + epsilon)
        self.neg_objective_act = - tf.reduce_mean(tf.minimum(x=tf.multiply(proba_ratio,  self.advantage),
                                                             y=tf.multiply(self.clip_pr, self.advantage)))
        self.optimizer_act = tf.train.AdamOptimizer(a_lrate).minimize(self.neg_objective_act)

    def train_v(self, states, target_v):
        feed_dict = {self.v.x: states,
                     self.target_v: target_v}
        self.sess.run(self.optimizer_v, feed_dict)

    def evaluate_v(self, states, target_v):
        feed_dict = {self.v.x: states,
                     self.target_v: target_v}
        return self.sess.run(self.loss_v, feed_dict)

    def train_pol(self, states, old_act, old_log_probas, advantages):
        feed_dict = {self.pol.mlp.x: states,
                     self.pol.test_action: old_act,
                     self.advantage: advantages,
                     self.old_log_probas: old_log_probas}
        return self.sess.run(self.optimizer_act, feed_dict)

    def evaluate_pol(self, states, old_act, old_log_probas, advantages):
        feed_dict = {self.pol.mlp.x: states,
                     self.pol.test_action: old_act,
                     self.advantage: advantages,
                     self.old_log_probas: old_log_probas}
        return -self.sess.run(self.neg_objective_act, feed_dict)

def next_batch_idx(batch_size, data_set_size):
    batch_idx_list = np.random.choice(data_set_size, data_set_size, replace=False)
    for batch_start in range(0, data_set_size, batch_size):
        yield batch_idx_list[batch_start: min(batch_start+batch_size, data_set_size)]
def rollout(env, policy, render=False):
    obs = env.reset()
    done = False
    while not done:
        if render:
            env.render()
        act = policy(obs)
        nobs, rwd, done, _ = env.step(np.minimum(np.maximum(act, env.action_space.low), env.action_space.high))
        yield obs, act, rwd, done
        obs = nobs

def rollouts(env, policy, min_trans, render=False):
    keys = ['obs', 'act', 'rwd', 'done']
    paths = {}
    for k in keys:
        paths[k] = []
    nb_paths = 0
    while len(paths['rwd']) < min_trans:
        for trans_vect in rollout(env, policy, render):
            for key, val in zip(keys, trans_vect):
                paths[key].append(val)
        nb_paths += 1
    for key in keys:
        paths[key] = np.asarray(paths[key])
        if paths[key].ndim == 1:
            paths[key] = np.expand_dims(paths[key], axis=-1)
    paths["nb_paths"] = nb_paths
    return paths

def get_gen_adv(paths, v_func, dicount, lam):
    v_values = v_func(paths['obs'])
    gen_adv = np.empty_like(v_values)
    for rev_k, v in enumerate(reversed(v_values)):
        k = len(v_values) - rev_k - 1
        if paths['done'][k]:
            gen_adv[k] = paths['rwd'][k] - v_values[k]
        else:
            gen_adv[k] = paths['rwd'][k] + dicount * v_values[k+1] - v_values[k] + dicount * lam * gen_adv[k+1]
    return gen_adv, v_values

def main():
    seed = 0
    np.random.seed(seed)
    tf.set_random_seed(seed)
    #
    # env = gym.make('MountainCarContinuous-v0')
    env = RandomJumpEnv()
    env.seed(seed)

    #
    n_iter = 1000000
    min_trans_per_iter = 3200
    render_every = 100
    epochs_per_iter = 15
    exploration_sigma = 1.5
    discount = .99
    lam_trace = .95
    epsilon = .2
    batch_size = 32

    # mlp for v-function and policy and ppo
    sess = tf.Session()
    layer_sizes = [env.observation_space.shape[0]] + [64] * 2
    layer_activations = [tf.nn.relu] * (len(layer_sizes) - 2) + [tf.nn.tanh]
    policy = GaussianPolicy_MLP(sess, layer_sizes + [env.action_space.shape[0]], layer_activations + [tf.nn.tanh], exploration_sigma)
    v_mlp = MLP(layer_sizes + [1], layer_activations + [tf.identity])
    get_v = lambda obs: sess.run(v_mlp.out, {v_mlp.x: obs})
    ppo = PPO(sess, policy, v_mlp, epsilon)
    sess.run(tf.global_variables_initializer())
    #
    for it in range(n_iter):
        print('------------iter , {}, --------------'.format(it))
        if (it+1) % render_every == 0:
            render = True
        else:
            render = False

        paths = rollouts(env, policy=policy.predict, min_trans=min_trans_per_iter, render=render)
        print("averaged rewards: ", np.mean(paths['rwd']))

        # update the v-function
        for epoch in range(epochs_per_iter):
            gen_adv, v_values = get_gen_adv(paths, get_v, discount, lam_trace)
            v_targets = v_values + gen_adv

            #
            if epoch == 0:
                print('v-function: loss before updating is: ', ppo.evaluate_v(paths['obs'], v_targets))
            for batch_idx in next_batch_idx(batch_size, len(v_targets)):
                ppo.train_v(paths['obs'][batch_idx], v_targets[batch_idx])

        # update policy
        gen_adv, _ = get_gen_adv(paths, get_v, discount, lam_trace)
        print("advantages: std {0:.3f}, mean {1:.3f} min {2:.3f}, max {3:.3f}".format(np.std(gen_adv),
                                                                                      np.mean(gen_adv),
                                                                                      np.min(gen_adv),
                                                                                      np.max(gen_adv)))
        gen_adv = gen_adv / np.std(gen_adv)
        log_act_probas = policy.get_log_proba(paths['obs'], paths['act'])
        print("entropy: before update", sess.run(policy.entropy))
        print("policy: objective before updating ", ppo.evaluate_pol(paths['obs'], paths['act'], old_log_probas=log_act_probas, advantages=gen_adv))
        # updating
        for epoch in range(epochs_per_iter):
            for batch_idx in next_batch_idx(batch_size, len(gen_adv)):
                ppo.train_pol(paths['obs'][batch_idx],
                              paths['act'][batch_idx],
                              old_log_probas=log_act_probas[batch_idx],
                              advantages=gen_adv[batch_idx])

        print("entropy: before update", sess.run(policy.entropy))
        print("policy: objective before updating ", ppo.evaluate_pol(paths['obs'], paths['act'], old_log_probas=log_act_probas, advantages=gen_adv))
        #
        log_act_probas_new = policy.get_log_proba(paths['obs'], paths['act'])
        diff = np.exp(log_act_probas - log_act_probas_new)
        print("action ratio: min {0:.3f}, mean {1:.3f}, max {2:.3f}, std {3:.3f}".format(np.min(diff),
                                                                                         np.mean(diff),
                                                                                         np.max(diff),
                                                                                         np.std(diff)))

if __name__ == '__main__':
    main()