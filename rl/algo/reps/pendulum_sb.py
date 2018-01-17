"""
    Step-Based
    REPS algorithm on Pendulum Environment
    Gaussian Policy, Linear Mean.
    RBF Features
"""
import gym
# from gym.envs.classic_control.pendulum import PendulumEnv
from rl.env.random_jump import RandomJumpEnv
from rl.featurizer.rbf_featurizer import RBFFeaturizer
from rl.policy.value_estimator_np import ValueEstimatorNP
from rl.policy.gaussian_policy_np import GaussianPolicyNP
from rl.misc.dual_function import DualFunction, dual_function_gradient
from rl.misc.utilies import discount_norm_rewards, stable_log_exp_sum
from rl.misc.memory import ReplayMemory, Transition, FeaturesTransition
#
from functools import partial
import numpy as np
import itertools
import sys
import matplotlib.pyplot as plt
#
env = gym.make('Pendulum-v0')
# env = RandomJumpEnv()
print("Action space : ", env.action_space)
print("Action space low : ", env.action_space.low)
print("Action space high: ", env.action_space.high)
print("Observation space : ", env.observation_space)
print("Observation space low : ", env.observation_space.low)
print("Observation space high: ", env.observation_space.high)
#
def step_reps(env, featurizer, policy_fn, num_episodes, num_steps, num_samples, value_fn, eta_init, v_init, epsilon, gamma=1.0):
    for i_episodes in range(num_episodes):
        state = env.reset()
        memory = ReplayMemory(capacity=num_steps)
        for t in itertools.count():
            # TODO: ===========================
            # TODO: Need Sample of Actions
            # TODO: ===========================
            for i_samples in range(num_samples):
                action = policy_fn.predict_step(state)
            next_state, reward, done, _ = env.step(action)
            if type(next_state) is not float:
                next_state = next_state.reshape(len(next_state))
            #
            features = featurizer.transform(state)
            next_features = featurizer.transform(next_state)
            memory.push(features, action, next_features, reward)
            #
            if done or t >= num_steps:
                transitions = memory.sample()
                batch_data = FeaturesTransition(*zip(*transitions))
                eta_init, v_init, weights = dual_function_gradient(epsilon=epsilon, param_eta=eta_init,
                                                          param_v=v_init, data_set=batch_data)
                #
                value_fn.update_reps(new_param_v=v_init)
                policy_fn.update_step(weights=weights, Phi_features=batch_data.features, actions=batch_data.action)
                print("epsidoes {}, mean rewards: {}".format(i_episodes, np.mean(batch_data.reward)))
                break
            state = next_state
#
dim_featuries = 10
rbf_featurizer = RBFFeaturizer(env=env, dim_features=dim_featuries)
# rbf_featurizer.plot_examples()
#
policy_fn = GaussianPolicyNP(env, rbf_featurizer)
value_fn = ValueEstimatorNP(rbf_featurizer)
param_v = value_fn.param_v
param_eta = 3
epsilon = 2e-1
#
num_samples = 10
num_steps = 1000
num_episodes = 1000
#
step_reps(env=env,
          featurizer=rbf_featurizer,
          policy_fn = policy_fn,
          value_fn = value_fn,
          num_episodes = num_episodes,
          num_steps = num_steps,
          num_samples=num_samples,
          eta_init=param_eta,
          v_init=param_v,
          epsilon=epsilon,
          gamma=1.0)

