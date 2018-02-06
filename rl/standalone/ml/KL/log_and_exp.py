import matplotlib.pyplot as plt
import numpy as  np

plt.style.use("ggplot")
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

# log
# x = np.arange(0.001, 5, 0.001)
# y = x*np.log(x) - (x-1)
# ax.text(x=2, y=2, s=r"$y=x\log(x) - (x-1)$", horizontalalignment='center',fontsize=17,color='b')

# exp
# x = np.arange(-2, 2, 0.001)
# y = np.exp(x)
# ax.text(x=1, y=4, s=r"$y=e^x$", horizontalalignment='center',fontsize=17,color='b')

# affine function
# x = np.arange(-2, 2, 0.001)
# y = 2*x + 1
# ax.text(x=1, y=1, s=r"$y=2x+1$", horizontalalignment='center',fontsize=17,color='b')
#

# inverse
x1 = np.arange(0.01, 1, 0.001)
x2 = np.arange(-1, -0.01, 0.001)
y1 = 1 / x1
y2 = 1 / x2
ax.text(x=0.25, y=15, s=r"$\frac{1}{x}$", horizontalalignment='center',fontsize=20,color='b')

ax.plot(x1,y1)
ax.plot(x2,y2)
ax.set_xlabel("x")
ax.set_ylabel("y")

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

plt.show()