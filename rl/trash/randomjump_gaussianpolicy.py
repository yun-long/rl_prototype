from rl.env.random_jump import RandomJumpEnv
from rl.featurizer.rbf_featurizer import RBFFeaturizer
from rl.trash.gaussian_policy_np import GaussianPolicyNP
from rl.misc.dual_function import DualFunction
from rl.misc.memory import EpisodesStats
#
from functools import partial
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import itertools
import matplotlib.pyplot as plt
#
def stable_log_exp_sum(x, N=None):
    """
    y = np.log(np.sum(np.exp(x)) / len(x)) # not stable
      = np.max(x) + np.log(np.sum(np.exp(x - np.max(x))) / len(x)) # stable
    :param x:
    :return:
    """
    max_x = np.max(x)
    if N is None:
        y = max_x + np.log(np.sum(np.exp(x-max_x)))
    else:
        y = max_x + np.log(np.sum(np.exp(x-max_x)) / N)
    return y

def reps(env,featurizer, policy_fn, dual_fn, num_episodes, num_steps, num_samples, eta, v, epsilon, discounted_factor=1.0):
    stats = EpisodesStats(rewards=np.zeros(num_episodes))
    for i_episodes in range(num_episodes):
        state = env.reset()
        theta_samples = policy_fn.samples(num_samples)
        weights = []
        rewards_episode = []
        for i_samples in range(num_samples):
            theta_sample = theta_samples[i_samples]
            rewards = []
            features = []
            next_features = []
            actions = []
            for t in itertools.count():
                # take a step
                action = policy_fn.predict(state)
                next_state, reward, done, _ = env.step(action)
                # save
                # stats.rewards[i_episodes] += reward
                rewards.append(reward)
                features.append(featurizer.transform(state))
                next_features.append(featurizer.transform(next_state))
                actions.append(actions)
                #
                if t >= (num_steps-1):
                    N = len(rewards)
                    rewards = np.array(rewards)
                    # rewards = discount_norm_rewards(rewards, discounted_factor)
                    rewards = rewards.reshape((N,))
                    features = np.array(features).reshape((-1, N))
                    next_features = np.array(next_features).reshape((-1, N))
                    features_diff = next_features - features
                    # eta, v = dual_fn.update(rewards, features, next_features)
                    x0 = np.hstack([eta, v])
                    bounds = [(-np.inf, np.inf) for _ in x0]
                    bounds[0] = (0.00001, np.inf)

                    def dual_fn(rewards, features_diff, inputs):
                        param_eta = inputs[0]
                        param_v = inputs[1:]
                        td_error = rewards + np.dot(param_v, features_diff)
                        weights = td_error / param_eta
                        g = param_eta * epsilon + param_eta * stable_log_exp_sum(x=weights, N=len(rewards))
                        return g

                    # TODO: Error in here
                    def dual_grad(rewards, features_diff, inputs):
                        param_eta = inputs[0]
                        param_v = inputs[1:]
                        td_error = rewards + np.dot(param_v, features_diff)
                        Z = np.exp(td_error / param_eta)
                        grad_eta = epsilon + np.log(np.sum(Z) / len(Z)) - Z.dot(td_error) / (param_eta * np.sum(Z))
                        grad_theta = Z.dot(features_diff.T) / np.sum(Z)
                        return np.hstack([grad_eta, grad_theta])

                    opt_fn = partial(dual_fn, rewards, features_diff)
                    grad_opt_fn = partial(dual_grad, rewards, features_diff)
                    params_new, _, _ = fmin_l_bfgs_b(func=opt_fn,
                                                     x0=x0,
                                                     bounds=bounds,
                                                     fprime=grad_opt_fn,
                                                     maxiter=100,
                                                     disp=False)
                    eta = params_new[0]
                    # print(eta)
                    v = params_new[1:]
                    td_error = rewards.reshape((len(rewards),)) + np.dot(v, (next_features - features))
                    weights.append(np.exp(td_error / eta))
                    rewards_episode.append(np.mean(rewards))
                    break
        # print(eta)
        policy_fn.update(weights, theta_samples)
        stats.rewards[i_episodes] = np.mean(rewards_episode)
        print("Mean reward {}".format(np.mean(rewards_episode)))
    return stats

#
env = RandomJumpEnv()
print("Action space : ", env.action_space)
print("Action space low : ", env.action_space.low[0])
print("Action space high: ", env.action_space.high[0])
print("Observation space : ", env.observation_space)
print("Observation space low : ", env.observation_space.low[0])
print("Observation space high: ", env.observation_space.high[0])
#

num_featuries = 10
eta = 5
v = np.random.rand(num_featuries) / np.sqrt(num_featuries)
# print(v)
num_trials = 10
num_episodes = 50
num_steps = 1000
num_samples = 10
epsilon_coeffs = np.array([2]) * 1e-1
rbf_featurizer = RBFFeaturizer(env, num_featuries=num_featuries)
mean_rewards = np.zeros(shape=(num_trials, num_episodes, len(epsilon_coeffs)))
#
for i_epsilon, epsilon in enumerate(epsilon_coeffs):
    print("epsilon : ", epsilon)
    for i_trails in range(num_trials):
        print("Trials : ", i_trails)
        #
        policy_fn = GaussianPolicyNP(env, rbf_featurizer)
        # value_fn = ValueEstimatorNP(rbf_featurizer)
        dual_fn = DualFunction(eta_init=eta, v_init=v, epsilon=epsilon)
        #
        stats = reps(env=env,
                     featurizer=rbf_featurizer,
                     policy_fn=policy_fn,
                     dual_fn = dual_fn,
                     num_episodes=num_episodes,
                     num_steps = num_steps,
                     num_samples = num_samples,
                     eta = eta,
                     v = v,
                     epsilon=epsilon,
                     discounted_factor=0.95)
        mean_rewards[i_trails, :, i_epsilon] = stats.rewards


fig = plt.figure()
plt.hold("True")
ax = fig.add_subplot(111)
ax.set_xlabel("Iteration")
ax.set_ylabel("Average reward")
c = ['b', 'm', 'r']

for l in range(len(epsilon_coeffs)):
    r_mean = np.mean(mean_rewards[:,:,l], axis=0)
    r_std = np.std(mean_rewards[:,:,l], axis=0)
    plt.fill_between(range(num_episodes), r_mean - r_std, r_mean + r_std, alpha=0.3, color=c[l])
    plt.plot(range(num_episodes), r_mean, color=c[l], label='$\epsilon$ = ' + str(epsilon_coeffs[l]))
plt.legend(loc='lower right')
plt.show()
