import numpy as np

# number of T time step
T = 200
# number of samplers
H = 20
# number of actions
n = 1
# number of features
m = 10
#
Phi = np.ones(shape=(H, m, T))
# Q = np.ones(shape=(H,1,T))
Q = np.diag(np.ones(T*H))
A = np.ones(shape=(H,n,T))

# Theta = []
# for i in range(H):
#     theta = np.dot(Phi[i, :, :].T, np.diag(Q[i, :, 0]))
#     theta = np.dot(theta, Phi[i, :, :])
#     theta = np.dot(np.linalg.pinv(theta), Phi[i, :, :].T)
#     theta = np.dot(theta, np.diag(Q[i, :, 0]))
#     theta = np.dot(theta, A[i, :, :])
#     Theta.append(theta)

print(np.dot(Phi.T, Q))
# print(np.matmul(Phi.T, np.diag(Q)))
# print(np.linalg.pinv(np.dot(np.dot(Phi.T, np.diag(Q)), Phi)))
