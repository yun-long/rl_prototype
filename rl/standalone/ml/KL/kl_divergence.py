import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

from scipy.stats import norm
from scipy.integrate import quad

p  = lambda x : norm.pdf(x, 0, 2)
q  = lambda x : norm.pdf(x, 2, 2)
kl = lambda x : p(x) * np.log(p(x) / q(x))

range = np.arange(-10, 10, 0.001)

KL_int = quad(kl, -10, 10)
print("KL : ", KL_int)

# ============================ first plot
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.grid(True)
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
ax.set_xlim(-10, 10)
ax.set_ylim(-0.1, 0.25)
#
ax.text(-2.5, 0.17, 'p(x)', horizontalalignment='center',fontsize=17,color='b')
ax.text(4.5, 0.17, 'q(x)', horizontalalignment='center',fontsize=17,color='g')

plt.plot(range, p(range))
plt.plot(range, q(range))
# ============================ Second Plot
ax = fig.add_subplot(1,2,2)
ax.grid(True)
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
ax.set_xlim(-10,10)
ax.set_ylim(-0.1,0.25)

ax.text(3.5, 0.17, r'$DK_{KL}(p||q)$', horizontalalignment='center',fontsize=17,color='b')

ax.plot(range, kl(range))

ax.fill_between(range, 0, kl(range))

plt.savefig('KullbackLeibler.png',bbox_inches='tight')
plt.show()