import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from rl.misc.utilies import fig_to_image
import imageio
#
import matplotlib.gridspec as gridspec
#
plt.style.use('ggplot')
#

def alpha_fn(alpha=1.0):
    if alpha == 1.0:  # KL-divergence
        f = lambda x: x * np.log(x) - (x - 1)
    elif alpha == 0.0:  # Reverse KL
        f = lambda x: -np.log(x) + (x - 1)
    else:
        f = lambda x: ((np.power(x, alpha) - 1) - alpha * (x - 1)) \
                      / (alpha * (alpha - 1))
    return f

def wasserstein_distance(mu1, mu2, sigma1, sigma2):
    w2 = np.square(mu1-mu2) + (sigma1 + sigma2 - 2 * np.sqrt(np.sqrt(sigma2) *  sigma2 * np.sqrt(sigma2)))
    return w2

def exp_barrier_fun(beta=10.0, kl_target=0.1):
    fun = lambda x: np.exp(beta * (x - kl_target))
    return fun

def plot_p_q(i, ax, xrange, p, q, alpha):
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    # ax.set_xlim(xmin, xmax)
    # ax.set_ylim(-0.1, 0.25)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_title(r'$\alpha={}$'.format(alpha), rotation='vertical', x=-0.1, y=0.6)
    #
    ax.plot(xrange, p(xrange))
    ax.plot(xrange, q(xrange))
    # if i == 0:
    #     ax.set_title("p(x) and q(x)")

def plot_f(i, ax, xrange, f):
    # ax.grid(True)
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-1, 3)
    #
    # D_f = q(xrange) * f(p(xrange) / q(xrange))
    ax.plot(xrange, f(xrange))
    ax.fill(xrange, f(xrange), 'orange')
    # ax.set_title(r'$P(i)\log \frac{Q(i)}{P(i)}$')
    if i == 0:
        ax.set_title(r'$q(i)f(\frac{p(i)}{q(i)})$')

def plot_f_int(i, ax, xrange, kl_ints, pel):
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.set_xlim(mu_min, mu_max)
    ax.set_ylim(-0.01, 1)
    ax.set_yticks([0, 1])
    print(xrange)
    if len(xrange) > 1:
        ax.scatter(xrange, kl_ints, c=xrange)
    else:
        ax.scatter(xrange, kl_ints)
    ax.set_title(r'$\alpha={}$'.format(alpha))
    # ax.scatter(xrange, kl_ints)
    # ax.plot(xrange, pel)
    # if i == 0:
    #    ax.set_title(r'$D_{f}(p||q)$')

exp_fun = exp_barrier_fun(beta=100.0, kl_target=0.1)
#
xmin = -10
xmax = 10
mu_min = -2
mu_max = 2
xrange = np.arange(xmin, xmax, 0.1)
p_mu = np.arange(mu_min, mu_max, 0.1)
#
f_ints = []
exp_ints = []
#
alphas = [1, 0, 2, -1, 0.5, 5, -5]
for _ in range(len(alphas)):
    exp_ints.append([])
    f_ints.append([])

video_name = "f_{}.mp4".format('exp')

writer = imageio.get_writer("/Users/yunlong/Gitlab/rl_prototype/results/ml/kl/" + video_name, fps=5)

frames = []
#fig, axes = plt.subplots(len(alphas), 2, figsize=(10, 5), dpi=100)
#fig.set_size_inches(18.5, 10.5)
fig = plt.figure(figsize=(12, 6))
fig.set_size_inches(18, 9)
ax0 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
ax1 = plt.subplot2grid((3, 3), (0, 2))
ax2 = plt.subplot2grid((3, 3), (1, 0))
ax3 = plt.subplot2grid((3, 3), (1, 1))
ax4 = plt.subplot2grid((3, 3), (1, 2))
ax5 = plt.subplot2grid((3, 3), (2, 0))
ax6 = plt.subplot2grid((3, 3), (2, 1))
ax7 = plt.subplot2grid((3, 3), (2, 2))
axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7]
#

for mu in p_mu:
    p  = lambda x : norm.pdf(x, mu, 2)
    q  = lambda x : norm.pdf(x, 0, 2)
    #
    #
    for i, alpha in enumerate(alphas):
        if alpha == 'wgan':
            w = wasserstein_distance(mu, 0, 2, 2)
            f_int = [w]
        else:
            alpha_f = alpha_fn(alpha=alpha)
            f = lambda x : q(x) * alpha_f(p(x)/ q(x))
            f_int = quad(f, xmin, xmax)
        print(f_int[0])
        f_ints[i].append(f_int[0])
        exp_ints[i].append(exp_fun(f_int[0]))
        print("alpha : {}, \t \t f divergence : {}".format(alpha, f_int[0]))
        plot_p_q(i, ax=axes[0], xrange=xrange, p=p, q=q, alpha=alpha)
        # plot_f(i, ax=axes[i][1], xrange=xrange, f=f)
        plot_f_int(i, ax=axes[i+1], xrange=p_mu[:len(f_ints[i])], kl_ints=f_ints[i], pel=exp_ints[i])
        plt.suptitle("f-divergence demo".format(alpha))
        if i <= len(axes) - 5:
            plt.setp(axes[i+1].get_xticklabels(), visible=False)
    image = fig_to_image(fig)
    frames.append(image)
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #            ncol=2, models="expand", borderaxespad=0.)
    # plt.show()
    # plt.close("all")
    for i, ax in enumerate(axes):
        ax.clear()
    writer.append_data(image)

writer.close()
# imageio.mimsave("example.gif", frames, fps=10)
