"""Solve ChainEnv using vectorized implementation of REPS.

For a chain with 8 links, doing 500 steps per policy iteration,
convergence is observed in less than 20 iterations. However,
not every run results in convergence.
"""

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=6, suppress=True)
from gym.envs.toy_text.nchain import NChainEnv
# from rl.standalone.reps_boris.chain import ChainEnv
from rl.standalone.reps_boris.utils import gather_data, reps_dual, reps_pi, reps_pe, expected_return
#
ENV_SEED = 34982
AGENT_SEED = 144439
#
# Environment
# env = ChainEnv(5)
env = NChainEnv(5)
num_actions = env.action_space.n
num_states = env.observation_space.n
env.seed(ENV_SEED)
rnd = np.random.RandomState(AGENT_SEED)
A_space = np.arange(num_actions)

# Fixed features
phi_mat = np.eye(num_states)
phi = lambda s: phi_mat[s]

# Fixed parameters
eta = 4
n_steps = 500
n_iter = 30

# Initial parameters
Pi = np.ones((num_states, num_actions)) / num_actions
theta = np.random.randn(num_states)
n_th = num_states

# Policy iteration
Pi_all = [Pi]
for i in range(n_iter):
    # Gather data
    D = gather_data(env, lambda s: np.random.choice(A_space, p=Pi[s]), n_steps)
    # Minimize dual
    dual, A, sa_keys = reps_dual(D, eta, phi)
    #
    theta, lamda = reps_pe(dual, n_th, rnd)
    # Update policy
    Pi_new = reps_pi(env, Pi, A, sa_keys, eta, theta, lamda)
    # Remember new policy
    Pi_all.append(Pi_new)
    # Next step
    Pi = Pi_new
    print(Pi)

# Evaluate policies
# returns = np.array([expected_return(env, policy) for policy in Pi_all])
# plt.plot(returns)
# plt.show()
