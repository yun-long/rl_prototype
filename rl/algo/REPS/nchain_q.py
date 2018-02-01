from rl.featurizer.one_hot_featurizer import OneHotFeaturizer
from rl.policy.numpy.value_estimator import ValueEstimator
from rl.misc.memory import Transition, ReplayMemory
from gym.envs.toy_text.nchain import NChainEnv
from collections import defaultdict
from functools import partial
from scipy.optimize import minimize
import numpy as np
import itertools
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def process_transitions(episode_data, featurizer):
    n = defaultdict(float)
    r = defaultdict(float)
    features_diff = defaultdict(float)
    for t_i, transition in enumerate(episode_data):
        state_action = (transition.state, transition.action)
        r[state_action] += transition.reward
        n[state_action] += 1
        features_diff[state_action] += featurizer.transform(transition.next_state) \
                                       - featurizer.transform(transition.state)
    keys = []
    r_array = []
    features_diff_array = []
    for key, value in n.items():
        keys.append(key)
        r_array.append(r[key] / n[key])
        features_diff_array.append(features_diff[key] / n[key])
    return np.array(r_array), np.array(features_diff_array), keys

def optimize_dual_function(N, r, features_diff, init_eta, init_v, epsilon):
    init_v = init_v.reshape(len(init_v))
    x0 = np.hstack([init_eta, init_v])
    bounds = [(-np.inf, np.inf) for _ in x0]
    bounds[0] = (1e-9, np.inf)
    def dual_fn(inputs):
        param_eta = inputs[0]
        param_v = inputs[1:]
        advantage = r + np.dot(features_diff, param_v)
        max_adv = np.max(advantage)
        g = param_eta * epsilon + max_adv + epsilon * np.log(np.sum(1. * np.exp((advantage - max_adv)/param_eta) / N))
        Z = np.exp(advantage / param_eta)
        grad_eta = epsilon + max_adv + np.log(np.sum(1. * np.exp((advantage-max_adv) / param_eta) / N)) - Z.dot(advantage) / (param_eta * np.sum(Z))
        grad_v = Z.dot(features_diff) / np.sum(Z)
        return g, np.hstack([grad_eta, grad_v])
    opt_fn = partial(dual_fn)
    results = minimize(opt_fn, x0=x0, method="L-BFGS-B", jac=True, options={'disp':False}, bounds=bounds)
    eta = results.x[0]
    v = results.x[1:]
    weights = np.exp(r + np.dot(features_diff, v)) / eta
    return eta, v, weights


def reps_step_based(env, featurizer, num_episodes, param_eta0, param_v0, epsilon, discount_factor=1.0):
    num_actions = env.action_space.n
    num_states = env.observation_space.n
    q_dist = np.ones((num_states, num_actions)) / (num_states * num_actions)
    mu_dist = np.ones(num_states) / num_states
    policy_k = q_dist / mu_dist[:, None]
    episodes_reward = []
    weights_dict = defaultdict()
    for i_episode in range(num_episodes):
        episode = []
        state = env.reset()
        rewards = []
        for t in range(200):
            action_prob = policy_k[state]
            action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
            next_state, reward, done, _ = env.step(action)
            episode.append(Transition(state=state,
                                      action=action,
                                      next_state=next_state,
                                      reward=reward))
            rewards.append(reward)
            state = next_state
        # process sampled data
        r, features_diff, keys = process_transitions(episode, featurizer)
        # optimize dual function
        param_eta, param_v, weights = optimize_dual_function(N=len(episode),
                                                             r=r,
                                                             features_diff=features_diff,
                                                             init_eta=param_eta0,
                                                             init_v=param_v0,
                                                             epsilon=epsilon)
        for j, state_action in enumerate(keys):
            weights_dict[state_action] = weights[j]
        policy_k_1 = np.zeros_like(policy_k)
        for state in np.arange(num_states):
            for action_a in np.arange(num_actions):
                sum = 0
                for action_b in np.arange(num_actions):
                    sum = sum + (policy_k[state][action_b] * weights_dict[(state, action_b)])
                policy_k_1[state][action_a] = policy_k[state][action_a] * weights_dict[(state, action_a)] / sum
        policy_k = policy_k_1
        episodes_reward.append(np.mean(rewards))
        print("\rEpisode {}, Expected Return {}.".format(i_episode, np.mean(rewards)))
        print(policy_k)
    return episodes_reward

if __name__ == '__main__':
    #
    env = NChainEnv(n=5)
    print("Action Space : ", env.action_space)
    print("Observation Space : ", env.observation_space)
    # define featurizer for value function
    featurizer = OneHotFeaturizer(env=env)
    # define value function
    value_fn = ValueEstimator(featurizer=featurizer)
    #
    # initialization paramteres for dual function
    epsilon = 0.1
    param_eta0 = 5.0
    param_v0 = value_fn.param_v
    #
    num_trails = 10
    num_episodes = 50
    trails_reward = np.zeros((10, 50))
    for i_trail in range(10):
        reward = reps_step_based(env=env,
                                 featurizer=featurizer,
                                 num_episodes=num_episodes,
                                 param_eta0=param_eta0,
                                 param_v0=param_v0,
                                 epsilon=epsilon,
                                 discount_factor=1.0)
        trails_reward[i_trail, :] = reward
    #
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average reward")
    r_mean = np.mean(trails_reward, axis=0)
    r_std = np.std(trails_reward, axis=0)
    plt.fill_between(range(num_episodes), r_mean - r_std, r_mean + r_std, alpha=0.3)
    plt.plot(range(num_episodes), r_mean)
    plt.show()

