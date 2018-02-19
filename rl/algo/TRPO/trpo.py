from rl.policy.gp_fc import GPFC
import numpy as np
import gym

# define the environment
env = gym.make("Pendulum-v0").unwrapped
np.random.seed(1234)
env.seed(1234)
print("env action space : ", env.action_space)
print("env observation space : ", env.observation_space)

#
policy = GPFC(env, )