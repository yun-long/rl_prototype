"""Solve modified FrozenLake 8x8 using REPS."""

import numpy as np
import matplotlib.pyplot as plt
from rl.standalone.reps_boris.lake import FrozenLakeEnv
from rl.standalone.reps_boris.utils import expected_return, gather_data, reps_dual, \
  reps_pe, reps_pi

np.set_printoptions(precision=6, suppress=True)
#
ENV_SEED = 34982
AGENT_SEED = 144439
env = FrozenLakeEnv(map_name='4x4')

# ==== REPS
env.seed(ENV_SEED)
rnd = np.random.RandomState(AGENT_SEED)
A_space = np.arange(env.nA)
# Fixed features
phi_mat = np.eye(env.nS)
phi = lambda s: phi_mat[s]

# Fixed parameters
eta = 5
n_steps = 1000
n_iter = 15
n_th = env.nS

# Policy iteration
Pi_all = [np.ones((env.nS, env.nA)) / env.nA]
Pi = Pi_all[-1]
theta = None
for i in range(n_iter):
    # Gather data
    D = gather_data(env, lambda s: rnd.choice(A_space, p=Pi[s]), n_steps)
    # Minimize dual
    eta *= 0.7
    dual, A, sa_keys = reps_dual(D, eta, phi)
    if i == 0:
        theta = rnd.randn(env.nS)
    # theta, lamda = reps_pe(dual, i, theta)
    theta, lamda = reps_pe(dual, n_th, rnd)
    # Update policy
    Pi_all.append(reps_pi(env, Pi, A, sa_keys, eta, theta, lamda))
    Pi = Pi_all[-1]

# Evaluate policies
ret_reps = np.array([expected_return(env, policy) for policy in Pi_all])
_, ax = plt.subplots()
ax.plot(ret_reps)


# ==== Optimal policy for frozen lake



# ==== Visualize value function and policy
def plot_lake(ax, env, pi, V):
    ax.clear()
    ax.imshow(V.reshape(4, 4), cmap='gray', interpolation='none', clim=(0, 1))
    ax.set_xticks(np.arange(4) - .5)
    ax.set_yticks(np.arange(4) - .5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    a2uv = {0: (-1, 0), 1: (0, -1), 2: (1, 0), 3: (0, 1)}
    pi_mat = pi.reshape(4, 4)
    for y in range(4):
      for x in range(4):
        a = pi_mat[y, x]
        u, v = a2uv[a]
        plt.arrow(x, y, u * .3, -v * .3, color='m', head_width=0.1, head_length=0.1)
        plt.text(x, y, str(env.desc[y, x].item().decode()),
                 color='g', size=12, verticalalignment='center',
                 horizontalalignment='center', fontweight='bold')
    plt.grid(color='b', lw=2, ls='-')

V = theta @ phi_mat
pi = np.argmax(Pi, axis=1)
_, ax = plt.subplots()
plot_lake(ax, env, pi, V)

# TODO: Try using gamma

# ==== Value iteration
from rl.standalone.reps_boris.utils import value_iteration
# from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
# env = FrozenLakeEnv()
Vs, pis = value_iteration(env, 0.8, 20)
# V = Vs[-1] - 17.4657*12

fig, ax = plt.subplots()
plot_lake(ax, env, pis[-1], Vs[-1])

# TODO: Why does value iteration not work?
