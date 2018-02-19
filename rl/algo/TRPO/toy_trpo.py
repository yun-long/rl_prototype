from rl.policy.gp_fc import GPFC
from rl.value.value_fc import ValueFC
from rl.misc.utilies import discount_norm_rewards
#
import itertools
import numpy as np
import gym
import tensorflow as tf
from gym.envs.classic_control.cartpole import CartPoleEnv

# define the environment
env = gym.make("Pendulum-v0").unwrapped
np.random.seed(1234)
env.seed(1234)
print("env action space : ", env.action_space)
print("env observation space : ", env.observation_space)

#
policy = GPFC(env, learning_rate=1e-2)
value = ValueFC(env, learning_rate=1e-2)


def trpo(num_episodes, num_samples, policy, value):
    for i_ep in range(num_episodes):
        state = env.reset()
        ep_state = np.zeros(shape=(num_samples, env.observation_space.shape[0]))
        ep_action = np.zeros(shape=(num_samples, env.action_space.shape[0]))
        ep_adv = np.zeros(shape=(num_samples, 1))
        ep_rewards = np.zeros(shape=(num_samples, 1))
        for t in itertools.count():
            action = policy.predict(state)
            next_state, reward, done, _ = env.step(action)
            adv = reward + value.predict(next_state) - value.predict(state)
            ep_state[t, : ] = state
            ep_action[t, :] = action
            ep_adv[t, :] = adv
            ep_rewards[t, :] = reward
            if t >= num_samples-1:
                break
            state = next_state
        #
        ep_adv = discount_norm_rewards(ep_rewards, gamma=0.99)
        policy.train(state=ep_state, action=ep_action, adv=ep_adv)
        value.train(state=ep_state, val=ep_adv)
        print("i_ep: {}, rewards: {}".format(i_ep, np.mean(ep_rewards)))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    trpo(num_episodes=1000,
         num_samples=500,
         policy=policy,
         value=value)