from gym.envs.classic_control.cartpole import CartPoleEnv
from gym.envs.toy_text.nchain import NChainEnv
from rl.featurizer.rbf_featurizer import RBFFeaturizer
from rl.featurizer.one_hot_featurizer import OneHotFeaturizer
import numpy as np

env = CartPoleEnv()
# env = NChainEnv()
num_actions = env.action_space.n
print("Action space : ", env.action_space)
print("Observation space : ", env.observation_space)
f = RBFFeaturizer(env=env, dim_features=10)
# f = OneHotFeaturizer(env=env)

theta = np.ones((num_actions, f.num_features)) / np.sqrt(num_actions)
Mu = np.zeros(num_actions * f.num_features)
Cov = np.eye(num_actions*f.num_features) * 1e-1

num_samples = 100
num_roll_out = 200

def policy(state, theta):
    s_feat = f.transform(state)
    action_prob = np.dot(theta, s_feat)
    action_prob = np.exp(action_prob) / np.sum(np.exp(action_prob))
    action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
    return action

def sample(theta_sample):
    rewards = []
    data = []
    state = env.reset()
    while True:
        action = policy(state, theta_sample)
        next_state, reward, done, _ = env.step(action)
        data.append([state, action, next_state, reward])
        rewards.append(reward)
        if done:
            break
        state = next_state
    return rewards, data

for i_episode in range(1000):
    G, data = sample(theta)

    print("i_episode : {}, rewards : {}".format(i_episode, rewards))
    for s in range(env.observation_space.n):
        print("state : {}, action : {}".format(s, policy(state=s, theta=theta)))
