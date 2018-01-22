import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy.stats import multivariate_normal
from rl.misc.utilies import fig_to_image
import random

# Load and plot date
data = np.loadtxt("./gmm.txt")
plt.figure()
plt.plot(data[:,0], data[:,1], '.', markersize=5)
# plt.show()
# plt.close()

# define the model
num_gaussian = 4 # number of gassuain for fitting the data points
num_data = data.shape[0] # number of data points
#
mu = np.array([[0., 0.,], [1., 0.], [0., 1.], [1., 1.]])
cov = np.array([20.*np.identity(2)] for i in range(num_gaussian))
alpha = np.ones(num_gaussian) / num_data

# K-means init
K = 4
N = data.shape[0]
N_iterations = 20
# mean initialization
mean = np.zeros((K, data.shape[1]))
for i, n in enumerate(np.random.randint(low=0, high=N, size=4)):
    mean[i, :] = data[n, :]

# distances initialization
distances = np.empty([N, K])
#
frames = []
for iteration in range(0, N_iterations):
    # compute the distance between each data point and the mean of each cluster
    for k in range(0, K):
        distances[:, k] = np.linalg.norm(data-mean[k, :], axis=1)
    # assign data points to the clusters
    assignments = np.argmin(distances, axis=1)
    # Update the mean of each cluster
    fig = plt.figure()
    colors = ['r.', 'g.', 'b.', 'k.']
    for k in range(0, K):
        plt.plot(data[assignments==k,0], data[assignments==k, 1], colors[k], markersize=5)
        plt.plot(mean[k, 0], mean[k, 1], colors[k], markersize=20)
    plt.title("Iteration : {}".format(iteration))
    image = fig_to_image(fig)
    frames.append(image)
    plt.close()
    #
    for k in range(0, K):
        mean[k, :] = np.sum(data[assignments==k, :], axis=0) / np.sum(assignments==k)

imageio.mimsave("/Users/yunlong/Gitlab/rl_prototype/results/ml/k_means/k_means.gif", frames, fps=5)