import numpy as np
from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
import matplotlib.pyplot as plt
#
from rl.policy.gp_general import GPGeneral
from rl.featurizer.poly_featurizer import PolyFeaturizer
from rl.featurizer.rbf_featurizer import RBFFeaturizer
from rl.misc.utilies import discount_norm_rewards
#

env = Continuous_MountainCarEnv()
print("Action space : ", env.action_space)
print("Observation space : ", env.observation_space)

# featurizer = PolyFeaturizer(env=env)
featurizer = RBFFeaturizer(env)
gp = GPGeneral(env, featurizer)
learning_rate = 0.1

def sample(policy):
    A, Mu_s_theta, Sigma_s_theta, X_s, R = [],[],[],[],[]
    state = env.reset()
    while True:
        action, x_s, mu_s_theta, sigma_s_theta = policy.predict(state)
        next_state, reward, done, _ = env.step(action)
        A.append(action)
        Mu_s_theta.append(mu_s_theta)
        Sigma_s_theta.append(sigma_s_theta)
        X_s.append(x_s)
        R.append(reward)
        if done:
            break
        state = next_state
    return np.array(A), np.array(Mu_s_theta), np.array(Sigma_s_theta), np.array(X_s), np.array(R)

episode_rewards = []
for i_episode in range(500):
    A, Mu_s_theta, Sigma_s_theta, X_s, R = sample(policy=gp)
    print("\ni_episode : {}, rewards : {}".format(i_episode, np.sum(R)))
    episode_rewards.append(np.sum(R))
    R = discount_norm_rewards(rewards=R, gamma=0.99)
    for t in range(R.shape[0]):
        print("\r{}/{}".format(t,R.shape[0]),end="")
        G = R[t]
        gp.update_mu(A=A[t], Mu_s_theta=Mu_s_theta[t], Sigma_s_theta=Sigma_s_theta[t], X_s=X_s[t], R=G, alpha=learning_rate)
        # Update sigma 的时候有问题
        # gp.update_sigma(A=A[t], Mu_s_theta=Mu_s_theta[t], Sigma_s_theta=Sigma_s_theta[t], X_s=X_s[t], R=G, alpha=learning_rate)

state = env.reset()
while True:
    action, _, _, _ = gp.predict(state)
    next_state, reward, done, _ = env.step(action)
    env.render()
    if done:
        break
    state = next_state

fig = plt.figure()
plt.plot(episode_rewards)
plt.show()

