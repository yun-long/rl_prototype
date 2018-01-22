from rl.policy.numpy.gp_linear_mean import GPLinearMean
from rl.featurizer.rbf_featurizer import RBFFeaturizer
from rl.policy.numpy.value_estimator import ValueEstimator
from rl.misc.dual_function import dual_function_gradient
from rl.misc.utilies import discount_norm_rewards
from rl.misc.memory import ReplayMemory, FeaturesTransition
#
from functools import partial, namedtuple
from scipy.optimize import fmin_l_bfgs_b
from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
import itertools
import pandas as pd
import numpy as np

#
env = Continuous_MountainCarEnv()
print("Action space : ", env.action_space)
print("Action space low : ", env.action_space.low)
print("Action space high: ", env.action_space.high)
print("Observation space : ", env.observation_space)
print("Observation space low : ", env.observation_space.low)
print("Observation space high: ", env.observation_space.high)
#
#
dim_features = 5
rbf_featurizer = RBFFeaturizer(env, dim_features=dim_features)
# rbf_featurizer.plot_examples()
#
T = 200
num_episodes = 1500
H = 10
#
epsilon = 0.4
# number of sample rollouts
policy = GPLinearMean(env, rbf_featurizer)
value = ValueEstimator(rbf_featurizer)
v = np.squeeze(value.param_v)
eta = 5
#
for i_episodes in range(num_episodes):
    Weights = np.zeros((H,T))
    A = np.zeros((H,T))
    Phi = np.zeros((H,T,rbf_featurizer.num_features))
    Phi_next = np.zeros((H,T,rbf_featurizer.num_features))
    theta_samples = policy.sample_theta(num_samples=H)
    episode_rewards = []
    for h in range(H):
        state = env.reset()
        rewards = np.zeros(T)
        memory = ReplayMemory(capacity=T, type="FeaturesTransition")
        theta = theta_samples[h]
        for t in itertools.count():
            # action, noise = policy.sample_action(state)
            action = policy.predict_action(state, theta)
            next_state, reward, done, _ = env.step(action)
            #
            features = rbf_featurizer.transform(state)
            next_features = rbf_featurizer.transform(next_state)
            #
            A[h,t] = action
            Phi[h,t,:] = features
            rewards[t] = reward
            memory.push(features, action, next_features, reward)
            if done:
                next_state = env.reset()j
            if t >= (T-1):
                break
            state = next_state
        episode_rewards.append(np.sum(rewards))
        batch = memory.sample()
        transitions = FeaturesTransition(*zip(*batch))
        eta, v, weights = dual_function_gradient(epsilon=epsilon,
                                                 param_eta=eta,
                                                 param_v=v,
                                                 transitions=transitions)
        Weights[h,:] = weights
    policy.update_em2(Weights=Weights, Phi=Phi, A=A)
    print(np.mean(episode_rewards))





