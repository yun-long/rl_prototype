from rl.misc.utilies import print_envinfo
from rl.policy.gp_fc import GPFC
from gym.envs.classic_control.pendulum import PendulumEnv
import tensorflow as tf
import numpy as np
#
env = PendulumEnv().unwrapped
print_envinfo(env, disc_a=False, disc_s=False)

#
policy = GPFC(env, learning_rate=1e-2)
gamma = 0.9
batch_size = 30
num_episodes = 1000
T = 1000
#
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i_ep in range(num_episodes):
        state = env.reset()
        ep_s, ep_a, ep_r = [], [], []
        sum_reward = 0
        for t in range(T):
            action = policy.predict(state)
            next_state, reward, done, _ = env.step(action)
            ep_s.append(state)
            ep_a.append(action)
            ep_r.append((reward+8)/8)
            sum_reward += reward
            state = next_state
            #
            if (t+1) % batch_size == 0 or t == T-1:
                v_next = policy.value(state)
                discount_r = []
                for r in ep_r[::-1]:
                    v_next = r + gamma * v_next
                    discount_r.append(v_next)
                discount_r.reverse()
                ba, bs, br = np.vstack(ep_a), np.vstack(ep_s), np.vstack(discount_r)
                policy.update(states=bs,
                              actions=ba,
                              rewards=br)
                ep_s, ep_a, ep_r = [], [], []
        print("i_ep: {}, sum rewards: {}".format(i_ep, sum_reward))

