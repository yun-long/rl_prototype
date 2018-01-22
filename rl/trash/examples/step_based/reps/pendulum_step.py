"""
    Step-Based
    REPS algorithm on Pendulum Environment
    Gaussian Policy, Linear Mean.
    RBF Features
"""
import gym
# from gym.envs.classic_control.pendulum import PendulumEnv
from rl.featurizer.rbf_featurizer import RBFFeaturizer
from rl.trash.value_estimator_np import ValueEstimatorNP
from rl.trash.gaussian_policy_np import GaussianPolicyNP
#
import numpy as np

#
env = gym.make('Pendulum-v0')
# env = RandomJumpEnv()
print("Action space : ", env.action_space)
print("Action space low : ", env.action_space.low)
print("Action space high: ", env.action_space.high)
print("Observation space : ", env.observation_space)
print("Observation space low : ", env.observation_space.low)
print("Observation space high: ", env.observation_space.high)


def calculate_Return(rewards, gamma=0.95):
    R = np.zeros_like(rewards)
    R_sum = 0
    for t in reversed(range(0, rewards.shape[0])):
        R_sum = R_sum * gamma + rewards[t]
        R[t] = R_sum
    # R -= np.mean(R)
    # R = np.exp( (R - np.min(R)) / (np.max(R) - np.min(R)) )
    # beta = 3. / (np.max(rewards) - np.min(rewards))
    # weights = np.exp((rewards-np.max(rewards)) * beta)
    # weights = weights / sum(weights)
    return R
#
def step_reps(env, featurizer, policy_fn, num_episodes, num_steps, num_samples, value_fn, eta_init, v_init, epsilon, gamma=1.0):
    H = 20
    m = featurizer.num_features
    T = 200
    n = env.action_space.shape[0]
    for i_eposide in range(num_episodes):
        # samples
        Phi_n = np.zeros(shape=(T,H,m))
        A = np.zeros(shape=(T,H,n))
        Q = np.zeros(shape=(T,H,1))
        for h in range(H):
            state = env.reset()
            rewards = np.zeros(shape=(T,1))
            # rollouts
            for t in range(T):
                action = policy_fn.predict_step(state)
                next_state, reward, done, _ = env.step(action)
                #
                Phi_n[t, h, :] = featurizer.transform(state)
                A[t, h, :] = action
                rewards[t] = reward
                # Q[t, h, :] = reward
                #
                state = next_state
            Returns = calculate_Return(rewards)
            #
            Q[:, h, :] = Returns
        # norm_Q = np.zeros(shape=(H, T, 1))
        # for h in range(H):
        #     norm_Q[h, :, :] = (Q[h,:,:] - np.min(Q, axis=1)) / (np.max(Q, axis=1) - np.min(Q, axis=1))
        #
        policy_fn.update_step(weights=Q,
                              Phi=Phi_n,
                              Actions=A)
        print(np.sum(Q))

def episode_reps(env, featurizer, policy_fn, num_episodes, num_samples, num_steps):
    for i_episodes in range(num_episodes):
        # init_state = env.reset()
        theta_samples = policy_fn.samples_theta(num_samples)
        rewards = np.zeros(num_samples)
        for i_samples in range(num_samples):
            state = env.reset()
            # state = init_state
            rewards_rollout = np.zeros(shape=num_steps)
            theta = theta_samples[i_samples]
            for t in range(num_steps):
                action = policy_fn.predict_episode(state, theta)
                next_state, reward, done, _ = env.step(action)
                rewards_rollout[t] = reward
                state = next_state
                if done:
                    break
            # Returns = calculate_Return(re)
            rewards[i_samples] = np.sum(rewards_rollout)
        beta = 5. / (np.max(rewards) - np.min(rewards))
        weights = np.exp((rewards-np.max(rewards)) * beta)
        weights = weights / sum(weights)
        policy_fn.update_episode(weights=weights, theta_samples=theta_samples)
        print("episode, ", i_episodes, np.mean(rewards))
        if i_episodes >= 700:
            obs = env.reset()
            while True:
                env.render()
                action = policy_fn.predict_step(obs)
                next_obs, reward, done, _ = env.step(action)
                obs = next_obs
                if done:
                    break

#
dim_featuries = 5
rbf_featurizer = RBFFeaturizer(env=env, dim_features=dim_featuries, beta=50)
# rbf_featurizer.plot_examples()
# time.sleep(1)
# plt.close()
#
policy_fn = GaussianPolicyNP(env, rbf_featurizer)
value_fn = ValueEstimatorNP(rbf_featurizer)
param_v = value_fn.param_v
param_eta = 3
epsilon = 5e-1
#
num_samples = 10
num_steps = 200
num_episodes = 10000
#
# step_reps(env=env,
#           featurizer=rbf_featurizer,
#           policy_fn = policy_fn,
#           value_fn = value_fn,
#           num_episodes = num_episodes,
#           num_steps = num_steps,
#           num_samples=num_samples,
#           eta_init=param_eta,
#           v_init=param_v,
#           epsilon=epsilon,
#           gamma=1.0)
episode_reps(env=env,
             featurizer=rbf_featurizer,
             policy_fn = policy_fn,
             num_episodes=num_episodes,
             num_samples = num_samples,
             num_steps=num_steps)
