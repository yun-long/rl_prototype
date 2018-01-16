from rl.env.random_jump import RandomJumpEnv
import numpy as np
import matplotlib.pyplot as plt
#
env = RandomJumpEnv()
#
class RBFFeaturizer(object):
    def __init__(self, env, num_features=10, beta=20):
        self.beta = beta
        self.obs_low = env.observation_space.low
        self.obs_high = env.observation_space.high
        self.norm_low = -1
        self.norm_high = 1
        self.num_features = num_features

    def normalizer(self, state):
        norm_state = (state - self.obs_low) / (self.obs_high - self.obs_low)
        norm_state = norm_state * 2 - 1
        return norm_state

    def transform(self,state):
        norm_state = self.normalizer(state)
        centers = np.array([i * (self.norm_high-self.norm_low) / (self.num_features-1) + self.norm_low for i in range(self.num_features)])
        phi = np.exp(-self.beta*(norm_state - centers) ** 2)
        return phi

    def plot_examples(self, show=True):
        N = 1000
        y_features = []
        x_features = []
        for state in np.linspace(self.obs_low, self.obs_high, N):
            x_features.append(state)
            features = self.transform(state)
            y_features.append(features)
        y_features = np.array(y_features)
        fig = plt.figure()
        for i in range(self.num_features):
            plt.plot(x_features, y_features[:, i])
            plt.hold(True)
        if show == True:
            plt.show()

class LinearPolicy(object):
    """
    Linear Policy
    """
    def __init__(self, env, learning_rate, num_features):
        self.num_outputs = env.action_space.shape[0]
        self.num_inputs = env.observation_space.shape[0]
        self.learning_rate = learning_rate
        self.num_features = num_features
        self.Mu_w = np.zeros(self.num_features)
        self.Sigma_w = np.eye(self.num_features) * 1e-2
        self.featurizer = RBFFeaturizer(env, num_features)
        self.moment = 0.9
        self.v_mu = 0
        self.v_sigma = 0

    def predict(self, state, theta):
        features = self.featurizer.transform(state).T
        action = np.dot(theta, features)
        action = np.atleast_2d(action)
        return action

    def update(self, theta_samples, rewards):
        Std_w = np.diag(self.Sigma_w)
        diff = theta_samples - self.Mu_w
        advantage = rewards - np.mean(rewards)
        #
        d_log_pi_Mu = diff * (1. / Std_w**2)
        d_log_pi_Std = diff * 2 / Std_w**3 - 1./Std_w
        d_log_pi_omega = np.hstack((d_log_pi_Mu, d_log_pi_Std))
        #
        G = np.dot(advantage, d_log_pi_omega) / num_samples
        # normalize variance gradient
        G_sigma = G[self.num_features: ]
        G_sigma = G_sigma / np.linalg.norm(G_sigma)
        G[self.num_features:] = G_sigma
        # Momentum Updates
        self.v_mu = self.moment * self.v_mu + self.learning_rate * G[:self.num_features]
        self.Mu_w += self.v_mu
        self.v_sigma = self.moment * self.v_sigma + self.learning_rate * G[self.num_features:]
        self.Sigma_w += self.v_sigma

    def samples(self, num_samples):
        theta = np.random.multivariate_normal(self.Mu_w, self.Sigma_w, num_samples)
        return theta

    def run_optimal_policy(self, num_run):
        for _ in range(num_run):
            obs = env.reset()
            while True:
                env.render()
                theta = policy.samples(num_samples=1)
                action = policy.predict(obs, theta)
                next_obs, rewards, done, _ = env.step(action)
                obs = next_obs
                if done:
                    # time.sleep(5)
                    break

num_features = 10
#
#
num_samples = 20
num_steps = 200
num_trials = 10
num_episodes = 500
# obs = env.reset()[:, None]
rewards_trials = np.zeros(shape=(num_trials, num_episodes))

for j in range(num_trials):
    policy = LinearPolicy(env=env,
                            learning_rate=1e-8,
                            num_features=num_features)
    for k in range(num_episodes):
        rewards_episode = []
        obs = env.reset()[:, None]
        while True:
            theta_samples = policy.samples(num_samples=num_samples)
            rewards = np.zeros(num_samples)
            for i, theta in enumerate(theta_samples):
                action = policy.predict(obs, theta)
                next_obs, reward, done, _ = env.step(action)
                rewards[i] = reward
            policy.update(theta_samples, rewards)
            obs = next_obs
            rewards_episode.append(np.mean(rewards))
            if done:
                break
        print("Trail - {0}, - Episode - {1}, - Mean Reward - {2}".format(j,k,np.mean(rewards_episode)))
        rewards_trials[j,k] = np.mean(rewards_episode)

# policy.run_optimal_policy(num_run=10)

fig = plt.figure()
plt.hold(True)
ax = fig.add_subplot(111)
ax.set_xlabel("Iteration")
ax.set_ylabel("Average Reward")
r_mean = np.mean(rewards_trials, axis=0)
r_std = np.std(rewards_trials, axis=0)
plt.fill_between(range(num_episodes), r_mean-r_std, r_mean+r_std, alpha=0.3)
plt.plot(range(num_episodes), r_mean)
plt.xlabel("number of episodes")
plt.ylabel("Mean rewards")
plt.show()





