import matplotlib.pyplot as plt
import numpy as  np

# plt.style.use("ggplot")

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

fig1 = plt.figure()
# f
x = np.arange(0.1, 3, 0.001)
f_kl = lambda x : x * np.log(x) - (x-1)
text1 = r"KL"
f_rkl = lambda x: -np.log(x) + (x-1)
text2 = r"Reverse KL"
f_pear = lambda x: 1. * np.square(x-1) / 2
text3 = r"Pearson$\chi^2$ "
f_ney = lambda x: np.square(x-1) / (2*x)
text4 = r"Neyman$\chi^2$"
f_hel = lambda x: 2 * np.square(np.sqrt(x) - 1)
text5 = r"Hellinger"
f_jen = lambda x: x * np.log(x) - (1+x) * np.log((1+x)/2.)
text6 = r"Jensen-Shannon"
f_gan = lambda x: x * np.log(x) - (1+x) * np.log(1+x)
text7 = r"GAN"
ax1 = fig1.add_subplot(2,1,1)
ax_process(fig1,ax1,
           [x,x,x,x,x,x,x],
           [f_kl(x), f_rkl(x), f_pear(x), f_ney(x), f_hel(x), f_jen(x), f_gan(x)],
           [text1, text2, text3, text4, text5, text6, text7],
           x_label='x',
           y_label="f(x)")
# f'
x = np.arange(0.2, 3, 0.001)
grad_f_kl = np.log(x)
text1 = r"KL"
grad_f_rkl = - 1./x + 1
text2 = r"Reverse KL"
grad_f_pear = x - 1
text3 = r"Pearson$\chi^2$"
grad_f_ney = - 1./ (2 * np.square(x)) + 1./2
text4 = r"Neyman$\chi^2$"
grad_f_hel = 2 - 2./ np.sqrt(x)
text5 = r"Hellinger"
grad_f_jen = -np.log((1+x)/(2*x)) - 1
text6 = r"Jensen-Shannon"
grad_f_gan = -np.log((1+x)/x)
text7 = r"GAN"
ax2 = fig1.add_subplot(2,1,2)
ax_process(fig1,ax2,
           [x,x,x,x,x,x,x],
           [grad_f_kl, grad_f_rkl, grad_f_pear, grad_f_ney, grad_f_hel, grad_f_hel, grad_f_gan],
           [text1, text2, text3, text4, text5, text6, text7],
           x_label='x',
           y_label="f'(x)")
plt.suptitle(r"generator functions $f(x)$ and its derivative $f'(x)$ of f-divergence")

fig2 = plt.figure()
# f*
y1 = np.arange(-2, 0.5, 0.01)
ff_kl = lambda y1: np.exp(y1) - 1
text1 = r"KL"
y2 = np.arange(-2, 0.5-0.01, 0.01)
ff_rkl = lambda y2: - np.log(1-y2)
text2 = r"Reverse KL"
y3 = np.arange(-1+0.01, 0.5, 0.01)
ff_pear = lambda y3 : np.square(y3+1) / 2 - 0.5
text3 = r"Pearson$\chi^2$"
y4 = np.arange(-2, 0.5-0.01, 0.01)
ff_ney = lambda y4 : -np.sqrt(1-2*y4) + 1
text4 = r"Neyman$\chi^2$"
y5 = np.arange(-2, 0.5-0.01, 0.01)
ff_hel = lambda y5 : 2 * y5 / (2-y5)
text5 = r"Hellinger"
ax1 = fig2.add_subplot(2,1,1)
ax_process(fig2,
           ax1,
           [y1,y2,y3,y4,y5],
           [ff_kl(y1), ff_rkl(y2), ff_pear(y3), ff_ney(y4), ff_hel(y5)],
           [text1, text2, text3, text4, text5],
           x_label='y',
           y_label=r"$f^{\ast}(y)$")

y1 = np.arange(-2, 0.5, 0.01)
grad_ff_kl = lambda y1: np.exp(y1)
text1 = r"KL"
y2 = np.arange(-2, 0.5-0.01, 0.01)
grad_ff_rkl = lambda y2: 1 / (1-y2)
text2 = r"Reverse KL"
y3 = np.arange(-1+0.01, 0.5, 0.01)
grad_ff_pear = lambda y3 : y3+1
text3 = r"Pearson$\chi^2$"
y4 = np.arange(-2, 0.5-0.01, 0.01)
grad_ff_ney = lambda y4 : 1 / np.sqrt(1-2*y4)
text4 = r"Neyman$\chi^2$"
y5 = np.arange(-2, 0.5-0.01, 0.01)
grad_ff_hel = lambda y5 : 4 / np.square(2-y5)
text5 = r"Hellinger"
ax2 = fig2.add_subplot(2,1,2)
ax_process(fig2,
           ax2,
           [y1,y2,y3,y4,y5],
           [grad_ff_kl(y1), grad_ff_rkl(y2), grad_ff_pear(y3), grad_ff_ney(y4), grad_ff_hel(y5)],
           [text1, text2, text3, text4, text5],
           x_label='y',
           y_label=r"$(f^{\ast})'(y)$")

plt.suptitle(r"convex conjugate $f^{\ast}(y)$ of generator functions and its derivative $(f^{\ast})'(y)$")
plt.show()