import matplotlib.pyplot as plt
import numpy as np
from rl.env.ball_throw import Pend2dBallThrowDMP
from rl.policy.gp_constant_mean import GPConstantMean
from rl.featurizer.rbf_featurizer import RBFFeaturizer

env = Pend2dBallThrowDMP()

num_dim = 10
num_samples = 25
max_iter = 100
num_trials = 10

alpha_coeffs = [0.1, 0.2, 0.4]
mean_reward = np.zeros(shape=(max_iter, num_trials, len(alpha_coeffs)))
#
rbf_featurizer = RBFFeaturizer
#
for l, alpha_coeff in enumerate(alpha_coeffs):
    for k in range(num_trials):
        policy = GPConstantMean(num_dim)
        for i in range(max_iter):
            # sample
            rewards = np.zeros(shape=num_samples)
            theta_samples = policy.sample_theta(num_samples)
            for j, theta_sample in enumerate(theta_samples):
                rewards[j] = env.getReward(theta=theta_sample)
            mean_reward[i, k, l] = np.mean(rewards)
            #
            advantage = rewards - np.mean(rewards)
            #
            policy.update_pg(alpha_coeff, theta_samples, advantage)
            #
            buf = ('Alpha ' + str(alpha_coeff) + ' - Trial ' + str(k) + ' - Iteration ' +
                   str(i) + ' - Mean reward ' + str(mean_reward[i, k, l]))

            print(buf)
            # terminal conditions
            if (i > 0 and abs(mean_reward[i, k, l]-mean_reward[i-1, k, l]) < 1e-3) or (i == max_iter-1):
                mean_reward[i:, k, l] = mean_reward[i, k, l] # fill missing iter
                break
        #
        if k == num_trials - 1:
            env.animate_fig(np.random.multivariate_normal(policy.Mu, policy.Sigma), alpha_coeff)

fig = plt. figure()
plt.hold(True)
ax = fig.add_subplot(111)
ax.set_xlabel('Iteration')
ax.set_ylabel('Average reward')
c = ['b', 'm', 'r']

for l in range(len(alpha_coeffs)):
    logRew = -np.log(-mean_reward)
    r_mean = np.mean(logRew[:,:,l],axis=1)
    r_std = np.std(logRew[:,:,l],axis=1)
    plt.fill_between(range(max_iter), r_mean - r_std, r_mean + r_std, alpha=0.3, color=c[l])
    plt.plot(range(max_iter), r_mean, color=c[l], label='$\lambda$ = ' + str(alpha_coeffs[l]))

plt.legend(loc='lower right')
plt.show()