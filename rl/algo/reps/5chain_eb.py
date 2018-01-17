import numpy as np
import gym
from collections import defaultdict
from scipy.optimize import fmin_l_bfgs_b
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot as plt

#
env = gym.make("NChain-v0")
# env = NChainEnv(n=5)
num_actions = env.action_space.n
num_states = env.observation_space.n
#
print("Number of actions : ", num_actions)
print("Number of states : ", num_states)

feature_type = "poly"
if feature_type == "one_hot":
    num_degree = num_states
elif feature_type == "poly":
    num_degree = 6
else:
    num_degree = 1

def featurizer(state, f_type=feature_type):
    if f_type == "poly":
        features = polynomial_features(state)
    elif f_type == "one_hot":
        features = one_hot_features(state)
    else:
        features = state
    return features

def epsilon_greedy_policy(Q, epsi=0.1):
    # print(Q)
    def policy_fn(observation):
        A = np.ones(num_actions, dtype=float) * epsi / num_actions
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsi)
        return A
    return policy_fn

def greedy_policy(Q):
    def policy_fn(observation):
        best_action = np.argmax(Q[observation])
        return best_action
    return policy_fn

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def stable_log_sum_exp(x, N=None):
    """
    y = np.log( np.sum(np.exp(x)) / len(x))  # not stable
      = np.max(x) + np.log(np.sum(np.exp(x-np.max(x)) / len(x))) # stable
    """
    a = np.max(x)
    if N is None:
        y = a + np.log(np.sum(np.exp(x-a)))
    else:
        y = a + np.log(np.sum(np.exp(x-a) / N))
    return y

def one_hot_features(state):
    state_one_hot = np.zeros(num_states)
    state_one_hot[state] = 1
    return state_one_hot


def polynomial_features(state):
    state_array = np.atleast_2d(state)
    features = PolynomialFeatures(degree=num_degree-1).fit_transform(state_array)
    return features[0]

# def rbf_features(state):

def samples(s, a, s_next, rs):
    tmp_r = defaultdict(lambda : 0)
    tmp_feature_diff = defaultdict(lambda : 0 )
    tmp_s_a = defaultdict(lambda : 0 )
    tmp_s = defaultdict(lambda : 0)
    for i in range(len(rs)):
        #
        state= s[i]
        action = a[i]
        next_state = s_next[i]
        reward = rs[i]
        #
        state_features = featurizer(state)
        next_state_features = featurizer(next_state)
        #
        tmp_feature_diff[(state, action)] += next_state_features - state_features
        tmp_s_a[(state, action)] += 1
        tmp_r[(state, action)] += reward
        tmp_s[state] += 1

    sum_rs = []
    sum_s_as = []
    sum_features_diff = []
    delta_Lambda_N = []
    q = defaultdict(lambda : 0)
    mu = defaultdict(lambda :0)
    for keys, values in tmp_s_a.items():
        delta_Lambda_N.append(tmp_feature_diff[keys] / tmp_s_a[keys])
        q[keys] = values / len(rs)
    for keys, values in tmp_s.items():
        mu[keys] = values / len(rs)

    for i in range(len(rs)):
        sum_rs.append(tmp_r[(s[i], a[i])])
        sum_s_as.append(tmp_s_a[(s[i], a[i])])
        sum_features_diff.append(tmp_feature_diff[(s[i], a[i])])

    def delta_sigma_dict(new_theta, state=None, action=None):
        dict_delta_theta = defaultdict(lambda : 0)
        dict_delta_N = []
        for keys, values in tmp_s_a.items():
            dict_delta_theta[keys] = (tmp_r[keys] + np.dot(tmp_feature_diff[keys], new_theta)) / values
            dict_delta_N.append( (tmp_r[keys] + np.dot(tmp_feature_diff[keys], new_theta)) / values)
        if (state == None) and (action==None):
            return np.array(dict_delta_N)
        else:
            return dict_delta_theta[(state, action)]
    return sum_s_as, sum_rs, sum_features_diff, delta_sigma_dict, q, mu, delta_Lambda_N

def bellman_error(sum_rs, sum_s_as, sum_features_diff):
    delta_Lambda = np.array(sum_features_diff) / np.array(sum_s_as)[:, None]
    def delta_sigma(theta):
        value = np.dot(sum_features_diff, theta)
        delta = sum_rs + value
        delta = delta / sum_s_as
        return delta
    return delta_sigma, delta_Lambda
# hyperparameters
epsilon = 2
#
numTrials = 5
maxIters = 100
meanReward = np.zeros(shape=(numTrials, maxIters))
s, a, s_next, rs = [], [], [], []
obs = env.reset()
reward_sum = 0
#
for m in range(numTrials):
    #
    param_theta_init = np.random.randn(num_degree) / np.sqrt(num_degree)
    param_eta_init = 10
    x0 = np.hstack([param_eta_init, param_theta_init])
    # initilization
    N_samples = 0
    num_episode = 0
    q_init = np.ones([num_states, num_actions]) / (num_actions)
    mu_init = np.ones(num_states) / num_states
    policy_init = q_init / mu_init[:, None]
    n = 0
    while True:
        # policy_init = q_init / mu_init[:, None]
        action_prob = q_init[obs]
        # action_prob = action_prob / mu_init[obs]
        # action_prob = policy_init[obs]
        # print(action_prob)
        # action = np.argmax(action_prob)
        action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
        next_obs, r, done, _ = env.step(action)
        #
        #
        reward_sum += r
        s.append(obs)
        a.append(action)
        s_next.append(next_obs)
        rs.append(r)
        N_samples += 1
        if done:
            # print(policy_init)
            meanReward[m,n] = np.mean(rs)
            # rs = np.array(rs)
            # rs = (rs - min(rs)) / (max(rs) - min(rs))
            sum_s_as, sum_rs, sum_features_diff, delta_sigma_dict, q, mu, delta_Lambda_N = samples(s, a, s_next, rs)
            delta_sigma, delta_Lambda = bellman_error(sum_rs, sum_s_as, sum_features_diff)
            #
            def dual_fun(inputs):
                param_eta = inputs[0]
                param_theta = inputs[1:]
                # print(delta_sigma_dict(param_theta))
                x = epsilon + 1. * delta_sigma(param_theta) / param_eta
                g = param_eta * stable_log_sum_exp(x, N=N_samples)
                return g

            def dual_faun_grad(inputs):
                param_eta = inputs[0]
                param_theta = inputs[1:]
                x = epsilon + 1. * delta_sigma(param_theta) / param_eta
                sum_share = np.sum(np.exp(x))
                sum_grad_theta = np.exp(x)[:, None] * delta_Lambda
                sum_grad_theta = np.sum(sum_grad_theta, axis=0)
                sum_grad_eta = np.sum(np.exp(x) * (1. * delta_sigma(param_theta) / (param_eta **2)))
                grad_theta = param_eta * sum_grad_theta / sum_share
                grad_eta = stable_log_sum_exp(x) - sum_grad_eta / sum_share
                return np.hstack([grad_eta, grad_theta])

            x0 = np.hstack([param_eta_init, param_theta_init])
            bounds = [(-np.inf, np.inf) for _ in x0]
            bounds[0] = (0., np.inf)
            params_new, _, _ = fmin_l_bfgs_b(func=dual_fun,
                                             x0=x0,
                                             bounds=bounds,
                                             fprime=dual_faun_grad,
                                             maxiter=100,
                                             disp=False)

            param_eta_init = params_new[0]
            param_theta_init = params_new[1:]
            # policy_tmp = policy_init
            for state in range(num_states):
                for action_a in range(num_actions):
                    sum = 0
                    weights = np.exp(1. * delta_sigma_dict(param_theta_init, state, action_a) / param_eta_init)
                    for action_b in range(num_actions):
                        sum += q[(state,action_b)] * np.exp(1. * delta_sigma_dict(param_theta_init, state, action_b) / param_eta_init)
                    q_init[state][action_a] = q[(state, action_a)] * weights / sum
                print("state : ", state, "actions : ", q_init[state])
                print("state : ", mu[state])
            print("Trails : ", m, "Episode : ", n, " - mean reward : ", reward_sum, "param eta : ", param_eta_init)
            #
            s, a, s_next, rs = [], [], [], []
            next_obs = env.reset()
            reward_sum = 0
            n += 1
            if n >= maxIters:
                break
        obs = next_obs

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel("Iteration")
ax.set_ylabel("Average reward")
r_mean = np.mean(meanReward, axis=0)
r_std = np.std(meanReward, axis=0)
plt.fill_between(range(maxIters), r_mean-r_std, r_mean+r_std,alpha=0.3)
plt.plot(range(maxIters), r_mean)
plt.show()

