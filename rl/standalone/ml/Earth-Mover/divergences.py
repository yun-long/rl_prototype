import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad
#
plt.style.use("ggplot")
#


def alpha_fn(alpha=1.0):
    if alpha == 1.0:  # KL-divergence
        # f = lambda x: x * np.log(x) - (x - 1)
        f = lambda x: x * np.log(x)
    elif alpha == 0.0:  # Reverse KL
        # f = lambda x: -np.log(x) + (x - 1)
        f = lambda x: -np.log(x)
    elif alpha == 'gan':
        f = lambda x:  x * np.log(x) - (1+x) * np.log((x + 1) / 2)
    else:
        f = lambda x: ((np.power(x, alpha) - 1) - alpha * (x - 1)) \
                      / (alpha * (alpha - 1))

    return f


def plot_p_q(ax, xrange, p, q):
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.plot(xrange, p(xrange), label='$p(x)$')
    ax.plot(xrange, q(xrange), label='$q(x)$')
    ax.legend()

def plot_f(ax, xrange, f, title, color='orange'):
    ax.grid(True)
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.plot(xrange, f(xrange), label=title)
    ax.fill(xrange, f(xrange), color, alpha=0.4)
    ax.legend()
    # ax.set_title(title)

p = lambda x : norm.pdf(x, -1, 2)

q = lambda x : norm.pdf(x, 0, 2)

xrange = np.arange(-2, 2, 0.01)

fig, axes = plt.subplots(2,2)
# plot p and q
plot_p_q(ax=axes[0][0], xrange=xrange, p=p, q=q)
# plot kl
f_1 = alpha_fn(alpha=1.0)
f = lambda x : q(x) * f_1(p(x)/ q(x))
f_int = quad(f, -2, 2)
print(f_int)
# f_int = quad(f, -10, 10)
plot_f(ax=axes[0][1], xrange=xrange, f=f, title=r'$D_{KL}(p || q)$')
# plot reverse kl
f_0 = alpha_fn(alpha=0.0)
f = lambda x : q(x) * f_0(p(x)/ q(x))
plot_f(ax=axes[0][1], xrange=xrange, f=f, title=r'$D_{KL}(q || p)$', color='blue')
#
f_gan = alpha_fn(alpha='gan')
f = lambda x : q(x) * f_gan(p(x)/ q(x))
# plot_f(ax=axes[1][0], xrange=xrange, f=f, title=r'$D_{kl}(p||\frac{p+q}{2})+D_{kl}(q||\frac{p+q}{2})$', color='blue')
plot_f(ax=axes[1][0], xrange=xrange, f=f, title=r'$D_{JS}(p||q)$', color='blue')
plt.show()
