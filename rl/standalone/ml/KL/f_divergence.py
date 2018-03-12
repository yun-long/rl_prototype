import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from rl.misc.utilies import fig_to_image
import imageio
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

def plot_p_q(i, ax, xrange, p, q, alpha):
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.1, 0.25)
    ax.set_title(r'$\alpha={}$'.format(alpha), rotation='vertical', x=-0.1, y=0.6)
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

def plot_f_int(i, ax, xrange, kl_ints):
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.01, 100)
    # ax.scatter(xrange, kl_ints, c=reversed(xrange))
    ax.scatter(xrange, kl_ints)
    if i == 0:
        ax.set_title(r'$D_{f}(p||q)$')
#
xmin = -20
xmax = 20
mu_min = -10
mu_max = 10
xrange = np.arange(xmin, xmax, 0.1)
p_mu = np.arange(mu_min, mu_max, 0.2)
#
f_ints = []
#
alphas = ['wgan', 1]
for _ in range(len(alphas)):
    f_ints.append([])

video_name = "f_{}.mp4".format('yy')
writer = imageio.get_writer("/Users/yunlong/Gitlab/rl_prototype/results/ml/kl/" + video_name, fps=5)
for mu in p_mu:
    p  = lambda x : norm.pdf(x, mu, 2)
    q  = lambda x : norm.pdf(x, 0, 2)
    #
    fig, axes = plt.subplots(len(alphas),2, figsize=(15,8), dpi=100)
    fig.set_size_inches(18.5, 10.5)
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
        print("alpha : {}, \t \t f divergence : {}".format(alpha, f_int[0]))
        plot_p_q(i, ax=axes[i][0], xrange=xrange, p=p, q=q, alpha=alpha)
        # plot_f(i, ax=axes[i][1], xrange=xrange, f=f)
        plot_f_int(i, ax=axes[i][1], xrange=p_mu[:len(f_ints[i])], kl_ints=f_ints[i])
        plt.suptitle("f-divergence demo".format(alpha))
    image = fig_to_image(fig)
    # frames.append(image)
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #            ncol=2, models="expand", borderaxespad=0.)
    # plt.show()
    plt.close("all")
    writer.append_data(image)

writer.close()
# imageio.mimsave("example.gif", frames, fps=10)
