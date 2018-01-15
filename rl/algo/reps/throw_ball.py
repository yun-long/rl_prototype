from matplotlib import pyplot as plt
import numpy as np
# from env.Pend2dBallThrowDMP import Pend2dBallThrowDMP
from rl.env.ball_throw import Pend2dBallThrowDMP
from scipy.optimize import minimize
from functools import partial

# parameters
env = Pend2dBallThrowDMP()
eta_init = 10 # eta
epsilon_coeffs = [0.1, 0.4, 0.9] # KL bound
numDim = 10
numSamples = 25
maxIter = 100
numTrials = 10

def dual_function(eps, rewards, eta):
    weights = np.exp(rewards/eta)
    g = eta * (eps) + eta * np.log(np.sum(weights/len(rewards)))
    dg = eps + np.log(np.sum(weights/len(rewards))) - weights.dot(rewards)/(eta*np.sum(weights))
    return g, dg

def optimize_dual_function(eps, rewards, x0):
    optfunc = partial(dual_function, eps, rewards)
    result = minimize(optfunc, x0, method="L-BFGS-B", jac=True, options={'disp': False}, bounds=[(0, None)])
    return result.x

def eval_policy(mu, sigma, numSamples):
    theta_samples = np.random.multivariate_normal(mu, sigma, numSamples)
    rewards = np.zeros(numSamples)
    for i in range(numSamples):
        rewards[i] = env.getReward(theta_samples[i, :])
    return theta_samples, rewards

def reps_update(rewards, eta_hat, theta_samples):
    weights = np.exp(rewards / eta_hat)
    print("weight shape", weights.shape)
    print("theta samples shape", theta_samples.shape)
    mu = weights.dot(theta_samples)/np.sum(weights)
    Z = (np.sum(weights)**2-np.sum(weights**2))/np.sum(weights)
    sigma = np.sum([weights[i]*(np.outer((theta_samples[i]-mu), (theta_samples[i]-mu))) for i in range(len(weights))], 0)/Z
    return mu, sigma

meanReward = np.zeros((maxIter, numTrials, len(epsilon_coeffs)))

for l, epsilon in enumerate(epsilon_coeffs):
    for k in range(numTrials):
        Mu_w = np.zeros(numDim)
        Sigma_w = np.eye(numDim) * 1e6
        theta_samples, rewards = eval_policy(Mu_w, Sigma_w, numSamples)
        for j in range(maxIter):
            # normalize rewards
            rewards_normalize = (rewards - min(rewards)) / (max(rewards) - min(rewards))
            # optimaize dual function
            eta_hat = optimize_dual_function(epsilon, rewards_normalize, eta_init)
            print(eta_init)
            # update the parameters of the Gaussian policy
            Mu_w, Sigma_w = reps_update(rewards_normalize, eta_hat, theta_samples)
            # take new samples
            theta_samples, rewards = eval_policy(Mu_w, Sigma_w, numSamples)
            #
            meanReward[j, k, l] = np.mean(rewards)
            buf = ('Lambda ' + str(epsilon) +' - Trial ' + str(k) +
                ' - Iteration ' + str(j) + ' - Mean Reward ' + str(meanReward[j,k, l]))
            print(buf)
            if (j > 0 and abs(meanReward[j,k,l]-meanReward[j-1, k, l]) < 1e-3 or j == maxIter):
                meanReward[j:, k, l] = np.mean(rewards)
                break
            # Update eta parameters, for dual function
            eta_init = eta_hat

        if k == numTrials -1:
            env.animate_fig(np.random.multivariate_normal(Mu_w, Sigma_w), epsilon)

fig = plt.figure()
plt.hold('on')
ax = fig.add_subplot(111)
ax.set_xlabel('Iteration')
ax.set_ylabel('Average reward')
c = ['b', 'm', 'r']

for l in range(len(epsilon_coeffs)):
    logRew = -np.log(-meanReward)
    r_mean = np.mean(logRew[:, :, l],axis=1)
    r_std = np.std(logRew[:, :, l],axis=1)
    plt.fill_between(range(maxIter), r_mean - r_std, r_mean + r_std, alpha=0.3, color=c[l])
    plt.plot(range(maxIter), r_mean, color=c[l], label='$\epsilon$ = ' + str(epsilon_coeffs[l]))
plt.legend(loc='lower right')
plt.show()