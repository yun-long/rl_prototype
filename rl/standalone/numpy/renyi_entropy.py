import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

def hartley_entropy(p):
    H_0 = np.log2(len(p))
    return H_0

def shannon_entropy(p):
    H_1 = - np.sum(p * np.log2(p))
    return H_1

def collision_entropy(p):
    H_2 = - np.log2(np.sum(p**2))
    return H_2

def min_entropy(p):
    max_p = np.max(p)
    H_inf = - np.log2(max_p)
    return H_inf

p = np.linspace(0, 1, 100)

h_0, h_1, h_2, h_inf = [], [], [], []
for p_i in p:
    P = np.array([p_i, 1-p_i])
    h_0.append(hartley_entropy(P))
    h_1.append(shannon_entropy(P))
    h_2.append(collision_entropy(P))
    h_inf.append(min_entropy(P))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(p, h_0, label=r"$H_0(p, 1-p)$")
ax.plot(p, h_1, label=r"$H_1(p, 1-p)$")
ax.plot(p, h_2, label=r"$H_2(p, 1-p)$")
ax.plot(p, h_inf, label=r"$H_{\infty}(p, 1-p)$")
ax.legend()
plt.show()
