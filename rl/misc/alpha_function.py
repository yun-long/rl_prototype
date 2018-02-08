"""
A summary of alpha function, its derivative, convex conjugate and derivative of convex conjugate
"""
import numpy as np
import matplotlib.pyplot as plt

def ax_process(fig, ax, x, y, text, x_label, y_label, xlim=None, ylim=None):
    for i, x_i in enumerate(x):
        y_i = y[i]
        ax.plot(x_i, y_i, label=text[i])

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()

    for side in ['bottom', 'right', 'top', 'left']:
        ax.spines[side].set_visible(False)

    # get width and height of axes object to compute
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    if xlim == None:
        xmin, xmax = ax.get_xlim()
    else:
        xmin, xmax = xlim
    if ylim == None:
        ymin, ymax = ax.get_ylim()
    else:
        ymin, ymax = ylim
    # manual arrowhead width and length
    hw = 1./40.*(ymax-ymin)
    hl = 1./40.*(xmax-xmin)
    lw = .5 # axis line width
    ohg = 0.3 # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

    # draw x and y axis
    ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw,
             head_width=hw, head_length=hl, overhang = ohg,
             length_includes_head= True, clip_on = False)
    #
    ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw,
             head_width=yhw, head_length=yhl, overhang = ohg,
             length_includes_head= True, clip_on = False)



# generator function and its convex conjugate for Reverse KL divergence
f_0 = lambda x : -np.log(x) + (x - 1)
ff_0 = lambda y : -np.log(1-y)
# generator function and its convex conjugate for KL divergence
f_1 = lambda x : x * np.log(x) - (x-1)
ff_1 = lambda y : np.exp(y) - 1

def alpha_fn(alpha):
    if alpha == 0:
        return f_0
    elif alpha == 1:
        return f_1
    f = lambda x : ((np.power(x, alpha) - 1) - alpha* (x-1) ) / (alpha * (alpha - 1))
    return f


def alpha_ffn(alpha):
    if alpha == 0:
        return ff_0
    elif alpha == 1:
        return ff_1
    ff = lambda y : np.power(1+(alpha-1)*y, alpha / (alpha - 1)) / alpha - 1 / alpha
    return ff



if __name__ == '__main__':
    alphas = [-10, -5, -1, 0, 1,  5, 10]
    x = np.arange(0+0.001, 2, 0.01)
    y = np.arange(-1, 1, 0.01)
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    for i, alpha in enumerate(alphas):
        fn = alpha_fn(alpha=alpha)
        ffn = alpha_ffn(alpha=alpha)
        ax_process(fig, ax1, [x], [fn(x)],["{}".format(alpha)], x_label="x", y_label="f(x)", xlim=(0, 2), ylim=(0,2))
        ax_process(fig, ax2, [y], [ffn(y)],["{}".format(alpha)], x_label="y", y_label=r"$f_{\alpha}^{\ast}(y)$", xlim=(-1,1), ylim=(-2,2))
    ax1.set_ylim(0, 2)
    ax2.set_ylim(-2,2)
    plt.suptitle(r"$f_\alpha (x)$ and $f_{\alpha}^{\ast}(y)$")
    plt.show()
