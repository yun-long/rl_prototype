import matplotlib.pyplot as plt
import numpy as  np

plt.style.use("ggplot")

# log
# x = np.arange(0.001, 5, 0.001)
# y = x*np.log(x) - (x-1)
# ax.text(x=2, y=2, s=r"$y=x\log(x) - (x-1)$", horizontalalignment='center',fontsize=17,color='b')

# exp
# x = np.arange(-2, 2, 0.001)
# y = np.exp(x)
# ax.text(x=1, y=4, s=r"$y=e^x$", horizontalalignment='center',fontsize=17,color='b')

#

# inverse
# x1 = np.arange(0.01, 1, 0.001)
# x2 = np.arange(-1, -0.01, 0.001)
# y1 = 1 / x1
# y2 = 1 / x2

def ax_process(ax, x, y, text):
    for i, x_i in enumerate(x):
        print(i)
        # if i==0:
        #     ax.plot(x_i, y[i],  label=text[i])
        ax.plot(x_i, y[i], label=text[i])
        # if i == 0:
        #     ax.text(x=2, y=-0.4, s=text[i], horizontalalignment='center',fontsize=15,color='b')
        # else:
        #     ax.text(x=3, y=5, s=text[i], horizontalalignment='center',fontsize=15,color='r')
    ax.set_xlabel("x")
    ax.set_ylabel("f'(x)")
    ax.legend()

    for side in ['bottom', 'right', 'top', 'left']:
        ax.spines[side].set_visible(False)

    # get width and height of axes object to compute
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

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

    ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw,
             head_width=yhw, head_length=yhl, overhang = ohg,
             length_includes_head= True, clip_on = False)

fig = plt.figure()
# affine function
# x1 = np.arange(0.01, 3, 0.001)
# y1 = x1 * np.log(x1) - (x1-1)
# x11 = np.arange(-2, -0.1, 0.001)
# y11 = 1 / x11
# text1 = r"$f(x)=x\log(x) - (x-1)$"
# x2 = np.arange(-2, 3, 0.001)
# y2 = np.exp(x2) - 1
# text2 = r"$f^{\ast}(y) = e^y - 1$"
# ax1 = fig.add_subplot(1,1,1)
# ax_process(ax1, [x1, x2], [y1, y2], [text1, text2])

# x = np.arange(0.1, 4, 0.001)
# y1 = x * np.log(x) - (x-1)
# text1 = r"KL"
# y2 = -np.log(x) + (x-1)
# text2 = r"Reverse KL"
# y3 = 1. * np.square(x-1) / 2
# text3 = r"Pearson$\chi^2$"
# y4 = np.square(x-1) / (2*x)
# text4 = r"Neyman$\chi^2$"
# y5 = 2 * np.square(np.sqrt(x) - 1)
# text5 = r"Hellinger"
# y6 = x * np.log(x) - (1+x) * np.log((1+x)/2.)
# text6 = r"Jensen-Shannon"
# y7 = x * np.log(x) - (1+x) * np.log(1+x)
# text7 = r"GAN"
x = np.arange(0.2, 4, 0.001)
y1 = np.log(x)
text1 = r"KL"
y2 = - 1./x + 1
text2 = r"Reverse KL"
y3 = x - 1
text3 = r"Pearson$\chi^2$"
y4 = - 1./ (2 * np.square(x)) + 1./2
text4 = r"Neyman$\chi^2$"
y5 = 2 - 2./ np.sqrt(x)
text5 = r"Hellinger"
y6 = -np.log((1+x)/(2*x)) - 1
text6 = r"Jensen-Shannon"
y7 = -np.log((1+x)/x)
text7 = r"GAN"
ax = fig.add_subplot(1,1,1)

ax_process(ax, [x, x, x, x, x, x, x], [y1, y2, y3, y4, y5, y6, y7], [text1, text2, text3, text4, text5, text6, text7])
plt.show()