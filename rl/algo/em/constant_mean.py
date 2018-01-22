from matplotlib import pyplot as plt
import numpy as np
from rl.env.ball_throw import Pend2dBallThrowDMP
from rl.policy.numpy.gp_constant_mean import GPConstantMean
import time

env = Pend2dBallThrowDMP()

numDim = 10
numSamples = 25
maxIter = 100
numTrials = 10

lambda_coeff = [3., 7., 25.]
# lambda_coeff = [3.]
meanReward = np.zeros((maxIter, numTrials, len(lambda_coeff)))

t = time.time()

for l in range(len(lambda_coeff)):
    for k in range(numTrials):
        policy = GPConstantMean(num_dim=numDim)
        for j in range(maxIter):
            rewards = np.zeros(numSamples)
            theta_samples = policy.sample_theta(num_samples=numSamples)
            for i in range(numSamples):
                rewards[i] = env.getReward(theta_samples[i, :])
            meanReward[j, k, l] = np.mean(rewards)
            # compute weights for EM
            beta = lambda_coeff[l] / (np.max(rewards) - np.min(rewards))
            weights = np.exp((rewards - np.max(rewards)) * beta)
            weights = weights / sum(weights)
            #
            policy.update_em(theta_samples, weights)
            #
            buf = ('Lambda ' + str(lambda_coeff[l]) + ' - Trial ' + str(k) +
                    ' - Iteration ' + str(j) + ' - Mean Reward ' + str(meanReward[j,k,l]))
            print(buf)

            if (j > 0 and abs(meanReward[j,k,l]-meanReward[j-1, k, l]) < 1e-3 or j == maxIter):
                meanReward[j:, k, l] = np.mean(rewards)
                break
        print( ' Elapsed : ' + str(time.time() - t))
        if k == numTrials -1:
            env.animate_fig(np.random.multivariate_normal(policy.Mu, policy.Sigma), lambda_coeff)

fig = plt.figure()
plt.hold('on')
ax = fig.add_subplot(111)
ax.set_xlabel('Iteration')
ax.set_ylabel('Average reward')
c = ['b', 'm', 'r']

for l in range(len(lambda_coeff)):
    logRew = -np.log(-meanReward)
    r_mean = np.mean(logRew[:,:,l],axis=1)
    r_std = np.std(logRew[:,:,l],axis=1)
    plt.fill_between(range(maxIter), r_mean - r_std, r_mean + r_std, alpha=0.3, color=c[l])
    plt.plot(range(maxIter), r_mean, color=c[l], label='$\lambda$ = ' + str(lambda_coeff[l]))

plt.legend(loc='lower right')
plt.show()