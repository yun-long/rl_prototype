import sys
if "../../../" not in sys.path:
    sys.path.append("../../../")

import gym
from rl.trash.examples.step_based.reinforce.reinforce import REINFORCE
# import matplotlib.pyplot as plt
import numpy as np

RENDER = False

env = gym.make("NChain-v0")
env.seed(1)
env = env.unwrapped

# print(env.action_space)
# print(env.observation_space.shape)
# print(env.observation_space.high)
# print(env.observation_space.low)

#center = (env.observation_space.high + env.observation_space.low) / 2.0
#print(center)

policy = REINFORCE(n_actions=env.action_space.n,
                   n_features=env.n,
                   learning_rate=0.02,
                   reward_decay=0.99,
                   outuput_graph=False)


for i_episode in range(1000):
    observation = env.reset()
    count = 0
    while True:
        count += 1
        # if RENDER:
        #     env.render()
        # print(np.array(observation).reshape(1,-1))
        obs_tmp = np.zeros(shape=env.n)
        obs_tmp[observation] = 1
        obs_feed = obs_tmp.reshape(1, -1)
        # print(obs_feed)
        action = policy.get_action(obs_feed)
        next_observation, reward, done, _ = env.step(action)
        policy.get_paths(obs_feed, action, reward)

        if count==100:
            ep_rs_rum = sum(policy.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_rum
            else:
                running_reward = running_reward * 0.99 + ep_rs_rum * 0.01

            print("\repisode : ", i_episode, " reward : ", int(running_reward))
            # if running_reward > -500:
            #     RENDER=True
            value = policy.train()
            break

        observation = next_observation
