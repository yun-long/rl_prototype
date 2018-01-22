from matplotlib import pyplot as plt
import numpy as np
from rl.env.ball_throw import Pend2dBallThrowDMP
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
        Mu_w = np.zeros(numDim)
        Sigma_w = np.eye(numDim) * 1e6
        for j in range(maxIter):
            rewards = np.zeros(numSamples)
            samples = np.random.multivariate_normal(Mu_w, Sigma_w, numSamples)
            for i in range(numSamples):
                rewards[i] = env.getReward(samples[i, :])
            meanReward[j, k, l] = np.mean(rewards)

            # compute weights for EM
            beta = lambda_coeff[l] / (np.max(rewards) - np.min(rewards))
            weights = np.exp((rewards - np.max(rewards)) * beta)
            weights = weights / sum(weights)

            # update mean and covariance
            Mu_w = np.dot(weights.T, samples)
            temp = samples - Mu_w
            temp1 = temp * weights[:, None]
            Sigma_w = np.dot(temp1.T, temp) + np.eye(numDim)

            buf = ('Lambda ' + str(lambda_coeff[l]) + ' - Trial ' + str(k) +
                    ' - Iteration ' + str(j) + ' - Mean Reward ' + str(meanReward[j,k,l]))
            print(buf)

            if (j > 0 and abs(meanReward[j,k,l]-meanReward[j-1, k, l]) < 1e-3 or j == maxIter):
                meanReward[j:, k, l] = np.mean(rewards)
                break
        print( ' Elapsed : ' + str(time.time() - t))
        if k == numTrials -1:
            env.animate_fig(np.random.multivariate_normal(Mu_w, Sigma_w), lambda_coeff)

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