"""Solve small fcliff using REPS."""

import matplotlib.pyplot as plt
from rl.standalone.reps_boris.fcliff import CliffWalkingEnv
from rl.standalone.reps_boris.utils import *

np.set_printoptions(precision=6, suppress=True)

ENV_SEED = 34982
AGENT_SEED = 14439

# TODO: Modify fclif to allow for different board sizes
env = CliffWalkingEnv()


# ==== REPS
env.seed(ENV_SEED)
rnd = np.random.RandomState(AGENT_SEED)
A_space = np.arange(env.nA)

# Map state s to auxiliary state x which excludes forbidden states
S_space = tuple(range(env.nS))
transient_states = tuple(np.where(env._cliff.ravel())[0]) + (env.nS - 1,)
X_space = tuple(s for s in S_space if s not in transient_states)
s_to_x = tuple((X_space.index(s) if s in X_space else None) for s in S_space)
phi_mat = np.eye(len(X_space))
phi = lambda s: phi_mat[s_to_x[s]]

# == Policy iteration
gamma = 1.0
n_iter = 10
n_steps = 30
# Initialization
th = rnd.randn(len(X_space))
Pi = np.ones((env.nS, env.nA)) / env.nA
# Keep track of primal and dual variables
Pi_all = [Pi]
th_all = []
lamda_all = []
for i in range(n_iter):
  # Data
  D = gather_data(env, lambda s: rnd.choice(A_space, p=Pi[s]), n_steps)
  # Critic
  eta = 5.0/(1.0 + i)
  dual, A, sa_keys = reps_dual(D, eta, phi, gamma=gamma)
  th, lamda = reps_pe(dual, th)
  # Actor
  # A_SA = np.zeros((env.nS, env.nA))
  A_SA = lamda * np.ones((env.nS, env.nA))
  A_SA[tuple(zip(*sa_keys))] = A(th)
  Pi_new = reps_pi(Pi, A_SA, eta)
  # Save data
  Pi_all.append(Pi_new)
  th_all.append(th)
  lamda_all.append(lamda)
  # Print out
  print("Iteration {}".format(i))
  # Next step
  Pi = Pi_new
ret_reps_all = [expected_return(env, policy) for policy in Pi_all]


# ==== Visualization
_, ax = plt.subplots()
ax.plot(ret_reps_all, 'o', label='reps_boris')
ax.legend()
ax.set_title('CliffWalking')
ax.set_xlabel('Iteration')
ax.set_ylabel('Expected reward')
plt.show()


# TODO: Value iteration doesn't seem to work with deterministic dynamics :-(
