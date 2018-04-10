import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
#
sns.set()
#

# path = '/Users/yunlong/Gitlab/rl_prototype/results/freps/NChain-v0/freps_-10.0_-4.0_-2.0_-1.0_0.0_0.5_1.0_2.0_3.0_5.0_10.0_data.csv'
path = '/Users/yunlong/Gitlab/rl_prototype/results/freps/Copy-v0/2018-03-20-02-33-13/freps_light_5_-10.0_-2.0_0.0_1.0_2.0_data.csv'
def plot_data(data_list):
    n_data = len(data_list)
    fig, axes = plt.subplots(nrows=1, ncols=n_data, figsize=(15, 3))
    for i, data in enumerate(data_list):
        ax = axes[i]
        # ax.set_yticks([2, 4])
        # ax = sns.tsplot(data=data, time='episode', value='reward', unit='trial', condition='alpha', ci='sd', ax=ax)
        # data_list.append(data.loc[data['alpha'].isin(alpha)])
        # print(data.loc[:, 'trial'])
        ax = sns.tsplot(data.loc['trial'].isin([0, 1, 2]))
        ax.set_xlabel('')
        ax.set_ylabel('')
    # plt.setp([a.get_xticklabels() for a in axes[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axes[1:]], visible=False)
    print(os.path.dirname(path))
    plt.savefig(os.path.join(os.path.dirname(path), 'result.png'))

if __name__ == '__main__':
    data = pd.read_csv(path)
    # alpha_list = [[-15.0, -10.0, 0.0, 1.0, 10.0],
    #              [-4.0, -2.0, 0.0, 1.0, 3.0, 5.0],
    #              [-1.0, 0.0, 0.5, 1.0, 2.0]]
    alpha_list = [[-15.0, -10.0, -2.0],
                   [0.0, 1.0, 2.0]]
    data_list = []
    for i, alpha in enumerate(alpha_list):
        data_list.append(data.loc[data['alpha'].isin(alpha)])
    plot_data(data_list=data_list)
    plt.show()
    # print(data.loc[data['alpha'].isin([-10.0, 0.0, 1.0, 10.0])])