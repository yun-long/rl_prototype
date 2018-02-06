from matplotlib import pyplot as plt
import numpy as np
from rl.env.ball_throw import Pend2dBallThrowDMP
from rl.policy.gp_constant_mean import GPConstantMean
from rl.misc.plot_rewards import plot_coeff_tr_ep_rs
import time

env = Pend2dBallThrowDMP()

numDim = 10
numSamples = 25
maxIter = 100
numTrials = 10

lambda_coeff = [3., 7., 25.]
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
            policy.update_wml(theta_samples=theta_samples,
                              weights=weights)
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
#
plot_coeff_tr_ep_rs(mean_rewards=meanReward, lambda_coeff=lambda_coeff, show=True)