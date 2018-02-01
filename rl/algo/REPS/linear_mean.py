"""
An implementation of REPS on Random Jumpy Environment, Gaussian Policy, Linear mean
"""
from rl.policy.numpy.gp_linear_mean import GPLinearMean
from rl.featurizer.rbf_featurizer import RBFFeaturizer
from rl.featurizer.non_featurizer import NoneFeaturizer
from rl.featurizer.poly_featurizer import PolyFeaturizer
from rl.policy.numpy.value_estimator import ValueEstimator
from rl.misc.dual_function import dual_function_gradient
from rl.misc.memory import ReplayMemory, FeaturesTransition
from rl.misc.plotting import plot_mean_rewards
from rl.env.random_jump import RandomJumpEnv
#
from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
from gym.envs.classic_control.pendulum import PendulumEnv
#
import itertools
import numpy as np
import sys
#
# env = Continuous_MountainCarEnv()
# env = PendulumEnv()
env = RandomJumpEnv()

print("Action space : ", env.action_space)
print("Action space low : ", env.action_space.low)
print("Action space high: ", env.action_space.high)
print("Observation space : ", env.observation_space)
print("Observation space low : ", env.observation_space.low)
print("Observation space high: ", env.observation_space.high)
#
# features for policy function
dim_features = 5
# policy_featurizer = RBFFeaturizer(env, dim_features=dim_features)
# rbf_featurizer.plot_examples()
policy_featurizer = NoneFeaturizer(env)
# policy_featurizer = PolyFeaturizer(env, degree=2)
# none_featurizer.plot_examples()
# features for value function
degree = 2
value_featurizer = PolyFeaturizer(env, degree=degree)
# poly_featurizer.plot_examples()

# Number of episodes
num_episodes = 100
#
epsilon = 0.5
# number of sample rollouts
policy = GPLinearMean(env, policy_featurizer)
value = ValueEstimator(value_featurizer)
v = np.squeeze(value.param_v)
eta = 10.0
#
episode_rewards = []
for i_episodes in range(num_episodes):
    A = []
    Phi = []
    rewards = []
    memory = ReplayMemory(capacity=500, type="FeaturesTransition")
    state = env.reset()
    for t in itertools.count():
        action = policy.predict_action(state)
        next_state, reward, done, _ = env.step(action)
        reward = np.atleast_1d(reward)
        # print(reward)
        #
        none_features = policy_featurizer.transform(state)
        Phi.append(none_features)
        A.append(action)
        rewards.append(reward)
        #
        poly_features = value_featurizer.transform(state)
        poly_next_features = value_featurizer.transform(next_state)
        #
        memory.push(poly_features, action, poly_next_features, reward)
        if t >= 500-1:
            break
        if done:
            next_state = env.reset()
        state = next_state
    episode_rewards.append(np.mean(rewards))
    batch = memory.sample()
    transitions = FeaturesTransition(*zip(*batch))
    eta, v, weights = dual_function_gradient(epsilon=epsilon,
                                             param_eta=eta,
                                             param_v=v,
                                             transitions=transitions)
    # print(eta, v)
    A = np.array(A)
    Phi = np.array(Phi)
    policy.update_wml(Weights=weights, Phi=Phi, A=A)
    value.update(new_param_v=v)
    print("Episode : {}, Reward: {}".format(i_episodes, np.mean(rewards)))

# Optimal Policy Demo
import time
state = env.reset()
while True:
    action = policy.predict_action(state)
    next_state, rewards, done, _ = env.step(action)
    state = next_state
    env.render()
    if done:
        time.sleep(1)





