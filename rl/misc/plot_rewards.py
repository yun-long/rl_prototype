import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

def plot_ep_rewards(ep_rs, show=True):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Expected Rewards")
    ax.plot(ep_rs)
    if show:
        plt.show()

def plot_tr_ep_rs(tr_ep_rs, show=True):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Averaged Expected reward")
    r_mean = np.mean(tr_ep_rs, axis=0)
    r_std = np.std(tr_ep_rs, axis=0)
    plt.fill_between(range(tr_ep_rs.shape[1]), r_mean - r_std, r_mean + r_std, alpha=0.3)
    plt.plot(range(tr_ep_rs.shape[1]), r_mean)
    if show:
        plt.show()
    return fig

def plot_coeff_tr_ep_rs(mean_rewards, lambda_coeff, show=True):
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
