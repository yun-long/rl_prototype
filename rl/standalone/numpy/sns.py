import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from rl.misc.plot_rewards import plot_tr_ep_rs

x = np.linspace(0, 10, 100)
Y = []
for _ in range(10):
    noise = np.random.normal(0, 40, 100)
    y = 2 * x ** 2 + noise
    Y.append(y)
data = np.asarray(Y)
ax = sns.tsplot(data=data)
plot_tr_ep_rs(data, title='test')
