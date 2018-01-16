from rl.env.random_jump import RandomJumpEnv
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy.optimize import minimize
import pandas as pd
#
class RBFFeaturizer(object):
    def __init__(self, env, num_features=10, beta=20):
        self.beta = beta
        self.obs_low = env.observation_space.low
        self.obs_high = env.observation_space.high
        self.norm_low = -1
        self.norm_high = 1
        self.num_features = num_features

    def normalizer(self, state):
        norm_state = (state - self.obs_low) / (self.obs_high - self.obs_low)
        norm_state = norm_state * 2 - 1
        return norm_state

    def transform(self,state):
        norm_state = self.normalizer(state)
        centers = np.array([i * (self.norm_high-self.norm_low) / (self.num_features-1) + self.norm_low for i in range(self.num_features)])
        phi = np.exp(-self.beta*(norm_state - centers) ** 2)
        return phi

    def plot_examples(self, show=True):
        N = 1000
        y_features = []
        x_features = []
        for state in np.linspace(self.obs_low, self.obs_high, N):
            x_features.append(state)
            features = self.transform(state)
            y_features.append(features)
        y_features = np.array(y_features)
        fig = plt.figure()
        for i in range(self.num_features):
            plt.plot(x_features, y_features[:, i])
            plt.hold(True)
        if show == True:
            plt.show()

class LinearPolicy(object):
    def __init__(self, env, num_features):
        self.num_features = num_features
        self.Mu_w = np.zeros(self.num_features)
        self.Sigma_w = np.eye(self.num_features) * 1e-3
        self.featurizer = RBFFeaturizer(env, num_features)

    def predict(self, state, theta):
        features = self.featurizer.transform(state).T
        action = np.dot(theta, features)
        action = np.atleast_2d(action)
        return action

    def update(self, rewards, eta_hat, theta_samples):
        weights = np.exp(rewards / eta_hat)
        self.Mu_w = weights.dot(theta_samples) / np.sum(weights)
        # Z = (np.sum(weights)**2 - np.sum(weights**2)) / np.sum(weights)
        # self.Sigma_w = np.sum([weights[i]*(np.outer((theta_samples[i]-self.Mu_w), (theta_samples[i]-self.Mu_w))) \
        #                        for i in range(len(weights))], 0)/Z

    def samples(self, num_samples):
        theta = np.random.multivariate_normal(self.Mu_w, self.Sigma_w, num_samples)
        return theta

def stable_log_sum_exp(x, N=None):
    """
    y = np.log( np.sum(np.exp(x)) / len(x))  # not stable
      = np.max(x) + np.log(np.sum(np.exp(x-np.max(x)) / len(x))) # stable
    """
    a = np.max(x)
    if N is None:
        y = a + np.log(np.sum(np.exp(x-a)))
    else:
        y = a + np.log(np.sum(np.exp(x -a) / N))
    return y

def dual_function(epsilon, rewards, eta):
    N = len(rewards)
    x = rewards / eta
    weights = np.exp(x)
    g = eta * epsilon + eta * stable_log_sum_exp(x, N)
    dg = epsilon + stable_log_sum_exp(x, N) - weights.dot(rewards) / (eta * np.sum(weights))
    return g, dg

def optimize_dual_function(eps, rewards, x0):
    optfunc = partial(dual_function, eps, rewards)
    result = minimize(optfunc, x0, method="L-BFGS-B", jac=True, options={'disp':False}, bounds=[(1e-10, np.inf)])
    return result.x

def save_parameters(Mu, Sigma):
    numbers = pd.Series()
    for i, mu in enumerate(Mu):
        numbers.loc[i] = mu
    for j, sigma in enumerate(np.diag(Sigma)):
        numbers.loc[len(Mu)+j] = sigma
    return numbers
#
env = RandomJumpEnv()
print("observation low : ", env.observation_space.low)
print("observation high : ", env.observation_space.high)
#
eta_init = 5
epsilon_coeffs = np.array([2, 4, 8]) * 1e-1 # KL bound
num_features = 10
num_trials = 10
num_episodes = 100
num_samples = 20
#
df = pd.DataFrame()
mean_rewards = np.zeros(shape=(num_trials, num_episodes, len(epsilon_coeffs)))
#
for l, epsilon in enumerate(epsilon_coeffs):
    for k in range(num_trials):
        policy = LinearPolicy(env=env,
                                num_features=num_features)
        eta_hat = eta_init
        for j in range(num_episodes):
            rewards_episode = []
            obs = env.reset()[:, None]
            T = 0
            theta_samples = policy.samples(num_samples=num_samples)
            numbers = save_parameters(policy.Mu_w, policy.Sigma_w)
            # print(numbers)
            df = df.append(numbers, ignore_index=True)
            rewards = []
            while True:
                T += 1
                # rewards = np.zeros(num_samples)
                rewards_tmp = []
                for i, theta in enumerate(theta_samples):
                    action = policy.predict(obs, theta)
                    next_obs, reward, done, _ = env.step(action)
                    rewards_tmp.append(reward)
                obs = next_obs
                rewards.append(rewards_tmp)
                if done or T >= 1000:
                    rewards = np.mean(rewards, axis=0).reshape(num_samples,)
                    # rewards_normalize = (rewards - min(rewards)) / (max(rewards) - min(rewards))
                    eta_hat = optimize_dual_function(epsilon, rewards, eta_hat)
                    policy.update(rewards, eta_hat, theta_samples)
                    rewards_episode.append(np.mean(rewards))
                    mean_rewards[k, j, l] = np.mean(rewards)
                    break
            print("epsilon == {0}, Trail == {1}, Episode == {2}, Mean Reward == {3}, eta == {4}".format(epsilon,k,j,np.mean(rewards_episode),eta_hat))
        if l ==0:
            df.to_csv("./output.csv")

fig = plt.figure()
plt.hold('on')
ax = fig.add_subplot(111)
ax.set_xlabel('Iteration')
ax.set_ylabel('Average reward')
c = ['b', 'm', 'r']

for l in range(len(epsilon_coeffs)):
    # logRew = -np.log(-mean_rewards)
    r_mean = np.mean(mean_rewards[:, :, l],axis=0)
    r_std = np.std(mean_rewards[:, :, l],axis=0)
    plt.fill_between(range(num_episodes), r_mean - r_std, r_mean + r_std, alpha=0.3, color=c[l])
    plt.plot(range(num_episodes), r_mean, color=c[l], label='$\epsilon$ = ' + str(epsilon_coeffs[l]))
plt.legend(loc='lower right')
plt.show()

