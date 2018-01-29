from rl.featurizer.one_hot_featurizer import OneHotFeaturizer
from rl.policy.numpy.value_estimator import ValueEstimator
from rl.misc.utilies import stable_log_exp_sum
from rl.misc.memory import Transition, ReplayMemory
# from rl.policy.numpy.discrete_policy import EpsilonGreedyPolicy
from gym.envs.toy_text.nchain import NChainEnv
from collections import defaultdict
from functools import partial
from scipy.optimize import minimize
import numpy as np

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


def reps_step_based(env, featurizer, num_episodes, param_eta, param_v, epsilon, discount_factor=1.0):
    num_actions = env.action_space.n
    num_states = env.observation_space.n
    q_dist = np.ones((num_states, num_actions)) / (num_states * num_actions)
    mu_dist = np.ones(num_states) / num_states
    policy_k = q_dist / mu_dist[:, None]
    # print("Initial state-action distribution \n{}".format(q_dist))
    # print("Initial state distribution \n{}".format(mu_dist))
    # print("Initial policy distribution \n{} ".format(policy))
    for i_episode in range(num_episodes):
        # sample transitions
        episode = []
        state = env.reset()
        rewards = []
        # print(policy)
        for t in range(400):
            action_prob = policy_k[state]
            action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
            next_state, reward, done, _ = env.step(action)
            episode.append(Transition(state=state,
                                      action=action,
                                      next_state=next_state,
                                      reward=reward))
            rewards.append(reward)
            if done:
                break
            state = next_state
        # process sampled data
        r, features_diff, keys = process_transitions(episode, featurizer)
        # optimize dual function
        param_eta, param_v, weights = optimize_dual_function(N=len(episode),
                                                             r=r,
                                                             features_diff=features_diff,
                                                             init_eta=param_eta,
                                                             init_v=param_v,
                                                             epsilon=epsilon)
        weights_dict = defaultdict()
        for j, state_action in enumerate(keys):
            weights_dict[state_action] = weights[j]
        policy_k_1 = np.zeros_like(policy_k)
        for state in np.arange(num_states):
            for action_a in np.arange(num_actions):
                sum = 0
                for action_b in np.arange(num_actions):
                    if (state, action_b) in weights_dict:
                        sum = sum + (policy_k[state][action_b] * weights_dict[(state, action_b)])
                if (state, action_a) in weights_dict:
                    policy_k_1[state][action_a] = policy_k[state][action_a] * weights_dict[(state, action_a)] / sum
        policy_k = policy_k_1
        print("\rEpisode {}, Expected Return {}".format(i_episode, np.mean(rewards)), end="")
    print(policy_k)

if __name__ == '__main__':
    #
    env = NChainEnv(n=5)
    print("Action Space : ", env.action_space)
    print("Observation Space : ", env.observation_space)
    # define featurizer for value function
    featurizer = OneHotFeaturizer(env=env)
    featurizer.print_examples()
    # define value function
    value_fn = ValueEstimator(featurizer=featurizer)
    # value_fn.plot_1D(env=env)
    #
    policy = None
    # initialization paramteres for dual function
    epsilon = 0.5
    param_eta = 10.0
    param_v = value_fn.param_v
    #
    reps_step_based(env=env,
                    featurizer=featurizer,
                    num_episodes=1000,
                    param_eta=param_eta,
                    param_v=param_v,
                    epsilon=epsilon,
                    discount_factor=1.0)



