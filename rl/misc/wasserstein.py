import numpy as np
import matplotlib.pyplot as plt
import ot
from ot.datasets import get_1D_gauss as gauss
from scipy.integrate import quad
#
plt.style.use('ggplot')
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
#
n = 100
n_traget = 41
#
x = np.arange(n, dtype=np.float64)
lst_m = np.linspace(20, 80, n_traget)
print(lst_m)

# gaussian distributions
f_kl = alpha_fn(alpha=1)
f_rkl = alpha_fn(alpha=0)
f_gan = alpha_fn(alpha='gan')

a = gauss(n, m=50, s=5)
p = lambda x: a[x]
B = np.zeros((n, n_traget))

kl_d = np.zeros(shape=n_traget)
reverse_kl = np.zeros(shape=n_traget)
gan = np.zeros(shape=n_traget)
for i, m in enumerate(lst_m):
    b = gauss(n, m=m, s=5)
    B[:, i] = b
    #
    q = lambda x: b[x]
    f_kld = lambda x: q(x) * f_kl(p(x) / q(x))
    f_rkld = lambda x: q(x) * f_rkl(p(x) / q(x))
    f_gand = lambda x: q(x) * f_gan(p(x) / q(x))
    for i_x in range(n):
        kl_d[i] += f_kld(i_x)
        reverse_kl[i] += f_rkld(i_x)
        gan[i] += f_gand(i_x)

# kl_d /= kl_d.max()
# reverse_kl /= reverse_kl.max()
# gan /= gan.max()

# loss matrix and normalization
print(x.reshape((n, 1)).shape)
M1 = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)), 'euclidean')
M1 /= M1.max()
M2 = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)), 'sqeuclidean')
M2 /= M2.max()

# Compute EMD for the different losses
d_emd = ot.emd2(a, B, M1)
d_emd2 = ot.emd2(a, B, M2)

# compute sinkhorn for the different losses
reg = 1e-2
d_sinkhorn = ot.sinkhorn2(a, B, M1, reg)
d_sinkhorn2 = ot.sinkhorn2(a, B, M2, reg)

#
fig = plt.figure(figsize=(12, 6))
ax0 = plt.subplot2grid(shape=(2, 2), loc=(0, 0))
ax1 = plt.subplot2grid(shape=(2, 2), loc=(1, 0))
ax2 = plt.subplot2grid(shape=(2, 2), loc=(0, 1), rowspan=2)
ax0.plot(x, a, 'b')
ax0.set_title("Source Distribution")
ax1.plot(x, B)
ax1.set_title("Target Dostrobutions")
plt.tight_layout()
#
# ax2.plot(lst_m, d_emd, label='Eucliden EMD')
# ax2.plot(lst_m, d_emd2, label='Squared Euclidean EMD')
ax2.plot(lst_m, d_sinkhorn, label='Wasserstein-1')
ax2.plot(lst_m, d_sinkhorn2, label='Wasserstein-2')
ax2.plot(lst_m, kl_d, label='KL divergence')
ax2.plot(lst_m, reverse_kl, '+', label='Reverse KL')
ax2.plot(lst_m, gan, label='GAN')
ax2.set_title('Distances')
ax2.set_ylim(0., 1)
ax2.legend()

plt.show()

