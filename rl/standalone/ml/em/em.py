import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from rl.misc.utilies import fig_to_image
import imageio
#
data = np.loadtxt("./gmm.txt")
plt.figure()
plt.plot(data[:,0], data[:,1], '.', markersize=5)
# plt.show()
N = data.shape[0]
K = 4
#
data_std = np.array([np.std(data[:,0]), np.std(data[:,1])])
data_mean = np.array([np.mean(data[:,0]), np.mean(data[:,1])])
#
cov = np.empty([K, 2, 2])
mu = np.empty([K, 2])
alpha = np.ones(K) / K
# print(alpha)
for k in range(K):
    cov[k, :, :] = np.array([[data_std[0], 1], [1, data_std[1]]])
    mu[k,:] = data_mean + np.random.random_sample(1)

n_iters = 50
responsibilities = np.empty([N, K])
log_like = np.empty(n_iters)
frames = []
for i_iter in range(n_iters):
    # E-Step
    for k in range(K):
        responsibilities[:,k] = alpha[k] * multivariate_normal.pdf(data, mu[k, :], cov[k, :, :])
    normalizer = np.sum(responsibilities, axis=1)
    responsibilities = responsibilities / normalizer[:, np.newaxis]
    # print(np.sum(responsibilities, axis=1))

    # Compute the log-likelihood
    log_like[i_iter] = np.sum(np.log(normalizer))

    # M-Step
    mu = np.dot(responsibilities.T, data)
    n = np.sum(responsibilities, axis=0)
    alpha = n / N
    cov = np.zeros([K, 2, 2])
    for k in range(K):
        mu[k, :] = mu[k, :] / n[k]
        cov[k, :, :] = np.dot(data.T, data*responsibilities[:, k, np.newaxis]) / n[k] - np.outer(mu[k, :], mu[k, :])

# # For plotting and save as gif
#     #
#     assigments = np.argmax(responsibilities, axis=1)
#     fig = plt.figure()
#     plt.title("Iteration " + str(i_iter))
#     color =['r.', 'g.', 'b.', 'k.']
#     x, y = np.mgrid[-2:4:.01, -2:5:.01]
#     pos = np.empty(x.shape + (2,))
#     pos[:, :, 0] = x
#     pos[:, :, 1] = y
#     for k in range(K):
#         plt.plot(data[assigments==k, 0], data[assigments==k, 1], color[k], markersize=5)
#         rv = multivariate_normal(mu[k, :], cov[k, :, :])
#         plt.contour(x, y, rv.pdf(pos))
#     image = fig_to_image(fig)
#     frames.append(image)
#     plt.close()
#
# imageio.mimsave("/Users/yunlong/Gitlab/rl_prototype/results/ml/em/em.gif", frames, fps=5)
#
# plt.figure()
# plt.xlabel("Iteration")
# plt.ylabel("Log-likelihood")
# plt.plot(log_like)
# plt.savefig("/Users/yunlong/Gitlab/rl_prototype/results/ml/em/log_likelihood.png")
# plt.show()

