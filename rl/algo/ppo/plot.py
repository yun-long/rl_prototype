import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#
sns.set()
#
path = '/Users/yunlong/Gitlab/rl_prototype/results/ppo/pendulum/data.csv'
data = pd.read_csv(path)
# g = sns.factorplot(x='episodes', y='rewards', hue='alphas', data=data, kind='strip')
fig = plt.figure(figsize=(15, 7))
ax1 = plt.subplot2grid(shape=(3, 3), loc=(0, 0), rowspan=2, colspan=2)
ax2 = plt.subplot2grid(shape=(3, 3), loc=(2, 0))
ax3 = plt.subplot2grid(shape=(3, 3), loc=(2, 1))
ax4 = plt.subplot2grid(shape=(3, 3), loc=(0, 2))
ax5 = plt.subplot2grid(shape=(3, 3), loc=(1, 2))
#
ax1 = sns.tsplot(data=data, time='episodes', value='rewards', unit='trials', condition='alphas', ax=ax1)
ax1.set_title("Expected reward")
ax1.set_ylabel('')
ax1.set_xlabel("Iteration")
ax2 = sns.tsplot(data=data, time='episodes', value='losses_c', unit='trials', condition='alphas', ax=ax2, legend=False)
ax2.set_title("Critic loss")
ax2.set_ylabel('')
ax2.set_xlabel('')
ax3 = sns.tsplot(data=data, time='episodes', value='losses_a', unit='trials', condition='alphas', ax=ax3, legend=False)
ax3.set_title("Actor loss")
ax3.set_ylabel('')
ax3.set_xlabel('')
ax4 = sns.tsplot(data=data, time='episodes', value='divergences', unit='trials', condition='alphas', ax=ax4, legend=False)
ax4.set_title("Policy divergence")
# ax4.set_ylim(0, 0.001)
ax4.set_ylabel('')
ax4.set_xlabel('')
ax5 = sns.tsplot(data=data, time='episodes', value='entropies', unit='trials', condition='alphas', ax=ax5, legend=False)
ax5.set_title("Entropy")
ax5.set_ylabel('')
ax5.set_xlabel('')
#
plt.tight_layout()
ax1.legend(bbox_to_anchor=(1.3, -0.5), loc='lower center', ncol=3, fancybox=True, shadow=True)
plt.show()

