"""Solve Chain using fREPS Light."""

import matplotlib.pyplot as plt
import seaborn as sns
from rl.standalone.reps_boris.chain import ChainEnv
from rl.standalone.reps_boris.utils import *
from rl.policy.discrete_policy import DistributionPolicy_V1
import gym
from gym.envs.algorithmic.copy_ import CopyEnv
#
np.set_printoptions(precision=6, suppress=True)
sns.set()

# Environment
# env = ChainEnv(8)
# S_space = tuple(range(env.nS))
# phi_mat = np.eye(len(S_space))
env_ID = 'Copy-v0'
env = CopyEnv(base=5)
S_space = tuple(range(env.observation_space.n))
phi_mat = np.eye(len(S_space))
# Parameters
params = dict(
  seeds=(147691, 43225801),
  alphas=(2.0,),
  # alphas=(-1.0, 0.0, 0.5, 1.0, 2.0),
  # alphas=(-4.0, -2.0, 0.0, 1.0, 3.0, 5.0),
  # alphas=(-10.0, 0.0, 1.0, 10.0),
  n_sim=10,
  n_iter=30,
  n_steps=800,
  etap=(15.0, 0.9, 0.1)
)
etap = params['etap']
etaf = lambda i: max(etap[0] * etap[1]**i, etap[2])
phi = lambda s: phi_mat[s]
policy = DistributionPolicy_V1(env)
# Simulation
sim_all = sim_f(env, etaf, phi, policy, **params)

# Returns
ret_all = {alpha: np.array([
  [expected_return(env, pi, policy) for pi in sim_traj['pi']]
  for sim_traj in sim_ensemble
]) for alpha, sim_ensemble in sim_all.items()}
ret_mat = np.stack([ret_all[alpha] for alpha in params['alphas']], axis=-1)

# Plot
fig, ax = plt.subplots()
alpha_names = [r'$\alpha = {:2.1f}$'.format(alpha)
               for alpha in params['alphas']]
ax = sns.tsplot(data=ret_mat, ax=ax, ci='sd', condition=alpha_names)
ax.plot(ret_mat[:, :, 0].T)
# ax.plot([0, params['n_iter']], [solve_lin_prog(env)[-1]]*2,
#         c='0.6', ls='dashed', label='optimal policy')
ax.set_title('Chain')
ax.set_xlabel('Iteration')
ax.set_ylabel('Expected reward')
plt.show()
# # Save data
# data_dir = 'submission/data/chain_light_fast/'
# fname = 'alpha_(-1)-2'
# outfile = data_dir + fname
# # Figure
# fig.savefig(outfile + '.pdf')
# # Metadata
# with open(outfile + '.json', 'w') as f:
#   json.dump(params, f)
# # Policies and returns
# np.save(outfile + '_sim.npy', sim_all)
# np.save(outfile + '_ret.npy', ret_all)
