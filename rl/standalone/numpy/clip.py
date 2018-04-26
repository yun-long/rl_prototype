import numpy as np
import matplotlib.pyplot as plt

def soft_clip(r, a, low, high):
    return np.minimum(np.maximum(r, low + a * (r - low)), high + a * (r - high))

A = 1
A_neg = -1
x = np.linspace(0, 2, 1000)
y_soft = soft_clip(x, a=-0.1, low=1.0, high=1.)

y1 = np.minimum(x*A, y_soft *A)
y2 = np.minimum(x*A_neg, y_soft * A_neg)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].plot(x, x * A, label='no clip')
axes[0].legend()
# axes[0].plot(x, y_soft)
axes[0].plot(x, y1, label='objecitve')
axes[1].plot(x, x*A_neg, label='objecitve')
axes[1].plot(x, y2, label='objecitve')
plt.show()
