"""
    Step-Based
    REPS algorithm on Random Jump Environment
    Gaussian Policy, Constant Mean.
    RBF Features
"""
# from gym.envs.classic_control.pendulum import PendulumEnv
from rl.env.random_jump import RandomJumpEnv
from rl.featurizer.rbf_featurizer import RBFFeaturizer
from rl.trash.value_estimator_np import ValueEstimatorNP
from rl.trash.gaussian_policy_np import GaussianPolicyNP
from rl.misc.dual_function import dual_function_gradient
#
from functools import partial
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import itertools

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

def reps(env, policy_fn, value_fn, num_episodes, num_steps,eta, v, discounted_factor=1.0):
    # stats = EpisodesStats(rewards=np.zeros(num_episodes))
    for i_episodes in range(num_episodes):
        state = env.reset()
        rewards = []
        features = []
        next_features = []
        actions = []
        for t in itertools.count():
            action = policy_fn.predict_step(state)
            next_state, reward, done, _ = env.step(action)
            # print(reward)
            #
            rewards.append(reward)
            features.append(value_fn.featurizer.transform(state))
            next_features.append(value_fn.featurizer.transform(next_state))
            actions.append(action)
            #
            if t >= num_steps:
                N = len(rewards)
                actions = np.array(actions)
                rewards = np.array(rewards)
                rewards = rewards.reshape((N,))
                features = np.array(features).reshape((-1, N))
                next_features = np.array(next_features).reshape((-1, N))
                features_diff = next_features - features
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
                v = params_new[1:]
                # print(eta)
                td_error = rewards.reshape((len(rewards),)) + np.dot(v, (next_features - features))
                weights = np.exp(td_error / eta)
                value_fn.update_reps(new_param_v = v)
                policy_fn.update_step(weights=weights, Phi=features.T, Actions=actions)
                break
            state = next_state
        print("mean rewards : ", np.sum(rewards))
        if i_episodes >= (num_episodes-1):
            obs = env.reset()
            while True:
                action = policy_fn.predict_step(obs)
                next_state, reward, done, _ = env.step(action)
                obs = next_state
                env.render()
                if done:
                    break


#
env = RandomJumpEnv()
# env = PendulumEnv()
print("Action space : ", env.action_space)
print("Action space low : ", env.action_space.low)
print("Action space high: ", env.action_space.high)
print("Observation space : ", env.observation_space)
print("Observation space low : ", env.observation_space.low)
print("Observation space high: ", env.observation_space.high)
#
num_features = 10
eta_init = 5
num_trials = 10
num_episodes = 50
num_steps = 1000
epsilon_coeffs = np.array([2]) * 1e-1
rbf_featurizer = RBFFeaturizer(env, dim_features=num_features)
# rbf_featurizer.plot_examples(show=True)
mean_rewards = np.zeros(shape=(num_trials, num_episodes, len(epsilon_coeffs)))
for i_epsilon, epsilon in enumerate(epsilon_coeffs):
    print("epsilon : ", epsilon)
    for i_trails in range(num_trials):
        print("Trials : ", i_trails)
        #
        policy_fn = GaussianPolicyNP(env, rbf_featurizer)
        value_fn = ValueEstimatorNP(rbf_featurizer)
        v_init = np.squeeze(value_fn.param_v)
        #
        stats = reps(env=env,
                     policy_fn=policy_fn,
                     value_fn=value_fn,
                     num_episodes=num_episodes,
                     num_steps=num_steps,
                     eta= eta_init,
                     v = v_init,
                     discounted_factor=1.0)

