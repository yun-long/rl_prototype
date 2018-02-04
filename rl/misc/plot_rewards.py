import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

def plot_episode_rewards(episode_rewards, show=True):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Expected Rewards")
    ax.plot(episode_rewards)
    if show:
        plt.show()

def plot_trail_episode_rewards(trail_episode_rewards, show=True):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Averaged Expected reward")
    r_mean = np.mean(trail_episode_rewards, axis=0)
    r_std = np.std(trail_episode_rewards, axis=0)
    plt.fill_between(range(trail_episode_rewards.shape[1]), r_mean - r_std, r_mean + r_std, alpha=0.3)
    plt.plot(range(trail_episode_rewards.shape[1]), r_mean)
    if show:
        plt.show()
    return fig

def plot_coeff_trail_episode_rewards(mean_rewards, lambda_coeff, show=True):
    fig = plt.figure()
    plt.hold('on')
    ax = fig.add_subplot(111)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average reward')
    c = ['b', 'm', 'r']

    for l in range(len(lambda_coeff)):
        logRew = -np.log(-mean_rewards)
        r_mean = np.mean(logRew[:,:,l],axis=1)
        r_std = np.std(logRew[:,:,l],axis=1)
        plt.fill_between(range(mean_rewards.shape[0]), r_mean - r_std, r_mean + r_std, alpha=0.3, color=c[l])
        plt.plot(range(mean_rewards.shape[0]), r_mean, color=c[l], label='$\lambda$ = ' + str(lambda_coeff[l]))
    plt.legend(loc='lower right')
    if show:
        plt.show()
