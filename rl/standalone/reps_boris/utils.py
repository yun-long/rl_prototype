import time
from functools import partial
import numpy as np
import scipy as sp
from scipy import optimize
import cvxopt as cvx
import json


def unit_vec(dim, k):
  """`k`-th standard basis vector in `dim`-dimensional space."""
  e = np.zeros(dim)
  e[k] = 1
  return e


# ================================ Plotting ================================= #
def plot_lake(ax, env, pi, V):
  ax.clear()
  ax.imshow(V.reshape(env.desc.shape))
  ax.set_xticks(np.arange(env.ncol) - .5)
  ax.set_yticks(np.arange(env.nrow) - .5)
  ax.set_xticklabels([])
  ax.set_yticklabels([])
  a2uv = {0: (-1, 0), 1: (0, -1), 2: (1, 0), 3: (0, 1)}
  pi_mat = pi.reshape(env.desc.shape)
  for y in range(env.nrow):
    for x in range(env.ncol):
      a = pi_mat[y, x]
      u, v = a2uv[a]
      ax.arrow(x, y, u * .3, -v * .3, color='m',
               head_width=0.1, head_length=0.1)
      ax.text(x, y, str(env.desc[y, x].item().decode()),
              color='K', size=12, verticalalignment='center',
              horizontalalignment='center', fontweight='bold')
  ax.grid(color='b', lw=2, ls='-')


def plot_flake(ax, env, mu, pi, v):
  """Plot value function and policy on top of FrozenLake."""
  # Exclude unreachable states
  unreachable_states = mu < 1e-8
  v_disp = np.copy(v)
  v_disp[unreachable_states] = np.nan
  # Display value function
  ax.clear()
  ax.imshow(v_disp.reshape(env.desc.shape))
  # Display policy
  ax.set_xticks(np.arange(env.ncol) - .5)
  ax.set_yticks(np.arange(env.nrow) - .5)
  ax.set_xticklabels([])
  ax.set_yticklabels([])
  a2uv = {0: (-1, 0), 1: (0, -1), 2: (1, 0), 3: (0, 1)}
  pi_mat = np.argmax(pi, axis=1).reshape(env.desc.shape)
  for y in range(env.nrow):
    for x in range(env.ncol):
      a = pi_mat[y, x]
      u, v = a2uv[a]
      ax.arrow(x, y, u * .3, -v * .3, color='m',
               head_width=0.1, head_length=0.1)
      ax.text(x, y, str(env.desc[y, x].item().decode()),
              color='k', size=12, verticalalignment='center',
              horizontalalignment='center', fontweight='bold')
  ax.grid(color='b', lw=2, ls='-')


def plot_fcliff(ax, env, mu, pi, v):
  # Exclude unreachable states
  unreachable_states = mu < 1e-8
  v_disp = np.copy(v)
  v_disp[unreachable_states] = np.nan
  # Display value function
  ax.clear()
  ax.imshow(v_disp.reshape(env.shape))
  # Display policy
  ax.set_xticks(np.arange(env.shape[1]) - .5)
  ax.set_yticks(np.arange(env.shape[0]) - .5)
  ax.set_xticklabels([])
  ax.set_yticklabels([])
  a2uv = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}
  pi_mat = np.argmax(pi, axis=1).reshape(env.shape)
  for y in range(env.shape[0]):
    for x in range(env.shape[1]):
      a = pi_mat[y, x]
      u, v = a2uv[a]
      ax.arrow(x, y, u * .3, -v * .3, color='m',
               head_width=0.1, head_length=0.1)
      # ax.text(x, y, str(env.desc[y, x].item().decode()),
      #         color='k', size=12, verticalalignment='center',
      #         horizontalalignment='center', fontweight='bold')
  ax.grid(color='b', lw=2, ls='-')


# =============================== MDP analysis ============================== #
def env_to_pr(env):
  """
  Extracts matrices P(s, a, s') and R(s, a, s') from env.P.
  
  :param env: DicreteEnv
  :return: (P, R) - dynamics and rewards
  """
  P = np.zeros((env.nS, env.nA, env.nS))
  R = np.zeros((env.nS, env.nA, env.nS))
  for s in env.P.keys():
    for a in env.P[s].keys():
      for tuple in env.P[s][a]:
        p, s_next, r, _ = tuple
        P[s, a, s_next] += p
        R[s, a, s_next] += r
  return P, R


def expected_return(env, pi):
  """
  Compute $\sum_{s,a,s'}{ \mu(s) \pi(a|s) P(s'|s,a) R(s,a,s') }$.

  :param env: DiscreteEnv
  :param pi: np.array of shape (env.nS, env.nA)
  :return: float
  """
  # Compute P(s'|s) under pi(a|s). P is implemented as array P[s,s']
  p, R = env_to_pr(env)
  P_pi = np.einsum('ij,ijk->ik', pi, p)
  # Find stationary distribution mu(s)
  vals, vecs = sp.linalg.eig(P_pi.T)
  unit_val_idx = np.argmax(np.real(vals))
  v = vecs[:, unit_val_idx]
  mu_0 = np.real(v / np.sum(v))
  # Check stationarity
  # print(mu_0, mu_0 @ P_pi)
  # Compute average reward under pi
  r = np.einsum('ijk,ijk->ij', R, p)
  rew = np.einsum('i,ij,ij->', mu_0, pi, r)
  return rew


def mu_under_pi(p, pi):
  P_pi = np.einsum('ij,ijk->ik', pi, p)
  # Find stationary distribution mu(s)
  vals, vecs = sp.linalg.eig(P_pi.T)
  unit_val_idx = np.argmax(np.real(vals))
  v = vecs[:, unit_val_idx]
  mu = np.real(v / np.sum(v))
  return mu


def rho_under_pi(p, pi):
  mu = mu_under_pi(p, pi)
  rho = mu[:, None] * pi
  return rho


# ======== Analytic solutions
def value_iteration(mdp, gamma, nIt, verbose=False):
  """
  Inputs:
      mdp: MDP
      gamma: discount factor
      nIt: number of iterations, corresponding to n above
  Outputs:
      (value_functions, policies)

  len(value_functions) == nIt+1 and len(policies) == n
  """
  if verbose:
    print("Iteration | max|V-Vprev| | # chg actions | V[0]")
    print("----------+--------------+---------------+---------")
  Vs = [np.zeros(mdp.nS)]
  pis = []
  for it in range(nIt):
    oldpi = pis[-1] if len(pis) > 0 else None
    Vprev = Vs[-1]
    Qs = np.zeros(mdp.nA)
    V = np.zeros(mdp.nS)
    pi = np.zeros(mdp.nS)

    for s in range(mdp.nS):
      for a in range(mdp.nA):
        Qs[a] = sum([t[0]*(t[2] + gamma*Vprev[t[1]]) for t in mdp.P[s][a]])
      V[s] = np.max(Qs)
      pi[s] = np.argmax(Qs)

    max_diff = np.abs(V - Vprev).max()
    nChgActions="N/A" if oldpi is None else (pi != oldpi).sum()
    if verbose:
      print("%4i      | %6.5f      | %4s          | %5.3f"%
            (it, max_diff, nChgActions, V[0]))
    Vs.append(V)
    pis.append(pi)
  return Vs, pis


def solve_lin_prog(env, gamma=1.0, mu0=0.0):
  """Find {mu, pi, v, lam} by solving an LP."""
  p, R = env_to_pr(env)
  nS, nA = env.nS, env.nA
  # Helper matrices to formulate optimization
  P = p.reshape((nS*nA, nS))
  r = np.einsum('ijk,ijk->ij', R, p).ravel()
  C = np.array([np.roll(np.r_[np.ones(nA), np.zeros((nS-1)*nA)], i*nA)
                for i in range(nS)]).T
  # Optimization problem
  x = cvx.Variable(nS*nA)
  objective = cvx.Maximize(r @ x)
  # Constraints
  # dyn_cons = C.T @ x == P.T @ x
  dyn_cons = C.T @ x == (1.0 - gamma) * mu0 + gamma * P.T @ x
  norm_cons = cvx.sum_entries(x) == 1.0
  pos_cons = x >= 0
  constraints = [dyn_cons, norm_cons, pos_cons]
  # Solve
  prob = cvx.Problem(objective, constraints)
  opts = dict(
    solver=cvx.ECOS,
    verbose=True,
    max_iters=200,
    abstol=1e-10,
    reltol=1e-8,
    feastol=1e-10
  )
  prob.solve(**opts)
  # State-action distribution rho(s,a)
  # rho = np.round(np.array(x.value).reshape((nS, nA)), 8)
  rho = np.array(x.value).reshape((nS, nA))
  rho[rho < 0.0] = 0.0
  rho /= np.sum(rho)
  rho = np.round(rho, 10)
  # Stationary state distribution mu(s)
  mu = C.T @ rho.ravel()
  # Optimal policy pi(a|s)
  pi = np.zeros((nS, nA))
  nz_idx = mu != 0
  pi[nz_idx] = rho[nz_idx] / mu[nz_idx, None]
  pi[~nz_idx] = np.ones(nA) / nA
  # Return J = lam
  lam = norm_cons.dual_value
  # Value function v(s)
  v = np.array(dyn_cons.dual_value).ravel()
  # Option 1: shift v and lam to satisfy lam = mu @ v
  # alpha = (lam - mu @ v) / (2.0 - gamma)
  pass
  # Option 2: shift v and lam to satisfy r @ x = mu @ v
  alpha = lam + ((1.0 - gamma) * mu0 - mu) @ v
  v = v + alpha
  # Option 3: shift v and lam to satisfy 0 = mu @ v (not implemented)
  lam = lam - alpha * (1.0 - gamma)
  return mu, pi, v, lam


# =================================== REPS ================================== #
def gather_data(env, pi, n_steps):
  """
  Collect `n_steps` samples (s, a, s', r) from `env` by following policy `pi`.

  :param env: DiscreteEnv
  :param pi: np.array of shape (nS, nA) - policy
  :param n_steps: int - number of steps
  :return: dict((s,a):(s',r)) - data set
  """
  D = {}
  s = env.reset()
  for i in range(n_steps):
    a = pi(s)
    s_next, rew, done, _ = env.step(a)
    if done:
      s_next = env.reset()
    if (s, a) in D:
      D[(s, a)].append((s_next, rew))
    else:
      D[(s, a)] = [(s_next, rew)]
    s = s_next
  return D


def adv(D, phi):
  D_sorted_keys = sorted(D)
  sa_multiplicities = [len(D[sa]) for sa in D_sorted_keys]
  R = np.array([
    sum((sn_and_rew[1] for sn_and_rew in D[sa])) / len(D[sa])
    for sa in D_sorted_keys
  ])
  DPhi = np.array([
    sum((phi(sn_and_rew[0])-phi(sa[0]) for sn_and_rew in D[sa])) / len(D[sa])
    for sa in D_sorted_keys
  ])
  A = lambda theta: R + DPhi @ theta
  return A, DPhi, sa_multiplicities, D_sorted_keys


def reps_dual(D, eta, phi):
  A, DPhi, sa_multiplicities, D_sorted_keys = adv(D, phi)
  # Define dual function
  def dual(theta):
    # Advantage
    Ath = A(theta)
    Amax = np.max(Ath)
    # Weights
    w = np.exp(-(Amax-Ath)/eta)
    # Dual and its derivative
    g = Amax + eta * np.log(np.average(w, weights=sa_multiplicities))
    dg = np.average(DPhi, axis=0, weights=w*sa_multiplicities)
    return g, dg
  return dual, A, D_sorted_keys


def reps_pe(dual, n_th, rnd):
  opts = dict(
    disp=False,
    iprint=2,
    maxiter=1000,
    ftol=1e-6
  )
  success = False
  while not success:
    th0 = rnd.randn(n_th)
    sol = sp.optimize.minimize(dual, th0, method='slsqp', jac=True,
                               options=opts)
    success = sol.success
    if not sol.success:
      print("Optimization failed!")
      print(sol)
  return sol.x, sol.fun


def reps_pi(env, pi, A, sa_keys, eta, th, lam):
  # Set advantage close to average for all (s,a) pairs
  A_SA = lam * np.ones((env.observation_space.n, env.action_space.n))
  # For visited (s,a) pairs, use computed advantage
  A_SA[tuple(zip(*sa_keys))] = A(th)
  # Update policy
  pi_new = np.copy(pi)
  pi_new *= np.exp(-(np.max(A_SA)-A_SA)/eta)
  pi_new /= np.sum(pi_new, axis=1, keepdims=True)
  return pi_new


# ==== Analytic
def repsa_step(env, pi0, gamma=1.0, mu0=0.0, eta=1.0):
  """Analytic policy improvement with REPS."""
  # Environment
  p, R = env_to_pr(env)
  nS, nA = env.nS, env.nA
  # Helper matrices
  P = p.reshape((nS*nA, nS))
  r = np.einsum('ijk,ijk->ij', R, p).ravel()
  C = np.array([np.roll(np.r_[np.ones(nA), np.zeros((nS-1)*nA)], i*nA)
                for i in range(nS)]).T
  # Initial state-action distribution
  rho0 = rho_under_pi(p, pi0)
  q = rho0.ravel()
  # Optimization problem
  x = cvx.Variable(nS*nA)
  objective = cvx.Maximize(r @ x - eta * cvx.sum_entries(cvx.kl_div(x, q)))
  # Constraints
  dyn_cons = C.T @ x == (1.0 - gamma) * mu0 + gamma * P.T @ x
  norm_cons = cvx.sum_entries(x) == 1.0
  pos_cons = x >= 0
  constraints = [dyn_cons, norm_cons, pos_cons]
  # Solve
  prob = cvx.Problem(objective, constraints)
  prob.solve(solver=cvx.ECOS, verbose=True, max_iters=200)
  # State-action distribution rho(s,a)
  rho = np.round(np.array(x.value).reshape((nS, nA)), 8)
  # Stationary state distribution mu(s)
  mu = C.T @ rho.ravel()
  # Optimal policy pi(a|s)
  pi = np.zeros((nS, nA))
  nz_idx = mu != 0
  pi[nz_idx] = rho[nz_idx] / mu[nz_idx, None]
  pi[~nz_idx] = np.ones(nA) / nA
  # Return J = lam
  lam = norm_cons.dual_value
  # Value function v(s)
  v = np.array(dyn_cons.dual_value).ravel()
  # Shift v and lam to satisfy r @ x = mu @ v
  alpha = r @ rho.ravel() - mu @ v
  v = v + alpha
  lam = lam - alpha * (1.0 - gamma)
  # Kappa
  k = np.array(pos_cons.dual_value).reshape((nS, nA))
  return mu, pi, v, lam, k


# ============================= Analytic fREPS ============================== #
def Fa(alpha=1.0):
  if alpha == 1.0:  # KL-divergence
    f = lambda x: x * np.log(x) - (x - 1)
    fp = lambda x: np.log(x)
    fcp = lambda y: np.exp(y)
    fc = lambda y: np.exp(y) - 1
  elif alpha == 0.0:  # Reverse KL
    f = lambda x: -np.log(x) + (x - 1)
    fp = lambda x: -1/x + 1
    fcp = lambda y: 1 / (1 - y)
    fc = lambda y: -np.log(1 - y)
  else:
    f = lambda x: ((np.power(x, alpha) - 1) - alpha * (x - 1)) \
                  / (alpha * (alpha - 1))
    fp = lambda x: (np.power(x, alpha-1) - 1) / (alpha - 1)
    fcp = lambda y: np.power(1 + (alpha - 1) * y, 1/(alpha - 1))
    fc = lambda y: 1/alpha * np.power(1 + (alpha - 1) * y, alpha/(alpha - 1)) \
                   - 1/alpha
  return f, fp, fcp, fc


def frepsa_step(env, pi0, f_div, eta):
  """Policy improvement with fREPS."""
  f, fp = f_div
  # Environment
  p, R = env_to_pr(env)
  nS, nA = env.nS, env.nA
  nX = nS*nA
  # State-action distribution to stay close to
  q = rho_under_pi(p, pi0).ravel()
  # Helper matrices
  P = p.reshape((nX, nS))
  r = np.einsum('ijk,ijk->ij', R, p).ravel()
  C = np.array([np.roll(np.r_[np.ones(nA), np.zeros((nS-1)*nA)], i*nA)
                for i in range(nS)]).T
  A = (C - P).T
  e = np.ones(nX)
  # Avoid division by zero in the argument of f-div
  nz_idx = q != 0
  y = np.ones(nX)
  def x_over_q(x, q):
    y[nz_idx] = x[nz_idx] / q[nz_idx]
    return y
  # Objective
  J = lambda x, q: -(r @ x - eta * q @ f(x_over_q(x, q)))
  DJ = lambda x, q: -(r - eta * fp(x_over_q(x, q)))
  # Constraints
  dyn_cons = lambda x: A @ x
  norm_cons = lambda x: np.sum(x) - 1
  cons = (
    {'type': 'eq', 'fun': dyn_cons, 'jac': lambda x: A},
    {'type': 'eq', 'fun': norm_cons, 'jac': lambda x: e}
  )
  bnds = np.array([np.zeros(nX), np.inf * np.ones(nX)]).T
  # Solve
  opts = dict(
    disp=False,
    iprint=2,
    maxiter=200,
    ftol=1e-6
  )
  sol = sp.optimize.minimize(J, q, jac=DJ, args=(q,), method='SLSQP',
                             constraints=cons, bounds=bnds, options=opts)
  if not sol.success:
    raise RuntimeError("Optimization failed!")
  # Extract policy
  rho = sol.x.reshape((nS, nA))
  mu = C.T @ rho.ravel()
  pi = np.zeros((nS, nA))
  nz_idx = mu != 0
  pi[nz_idx] = rho[nz_idx] / mu[nz_idx, None]
  pi[~nz_idx] = np.ones(nA) / nA
  return pi


# ==== Data collection
def pi_traj(env, alpha, n_iter, pi0, eta0):
  """Policy improvement trajectory pi_all = [pi_0, pi_1, ..., pi_n_iter]."""
  f, fp, _, _ = Fa(alpha)
  pi = np.copy(pi0)
  pi_all = [pi]
  for i in range(n_iter):
    if i % 10 == 0:
      print("Iteration {}".format(i))
    eta = 0.99**i * eta0
    pi = frepsa_step(env, pi, (f, fp), eta)
    pi_all.append(pi)
  return pi_all


def ret_ensemble(env, alpha, n_sim, n_iter, pi0, eta0):
  """Collect `n_sim` ret_traj's of length `n_iter` for a given `alpha`."""
  ret_all = []
  for k in range(n_sim):
    print("Rollout {}".format(k))
    pi_all = pi_traj(env, alpha, n_iter, pi0, eta0)
    ret_traj = [expected_return(env, pi) for pi in pi_all]
    ret_all.append(ret_traj)
  return np.array(ret_all)


def alpha_returns(env, alpha_all, n_sim, n_iter, pi0, eta0):
  """Iterate over `alpha_all` and call `ret_ensemble`."""
  ret_alpha_all = []
  for alpha in alpha_all:
    print("Alpha = {}".format(alpha))
    ret_all = ret_ensemble(env, alpha, n_sim, n_iter, pi0, eta0)
    ret_alpha_all.append(ret_all)
  return ret_alpha_all


# Function all-in-one
def pi_alpha_traj(env, pi0, eta0, alpha_all, n_sim, n_iter, **kwargs):
  print("Collect trajectories")
  pi_alpha_all = []
  for alpha in alpha_all:
    print("Alpha = {}".format(alpha))
    f, fp, _, _ = Fa(alpha)
    pi_alpha = []
    for k in range(n_sim):
      print("Rollout {}".format(k))
      pi = np.copy(pi0)
      pi_all = [pi]
      for i in range(n_iter):
        if i % 10 == 0:
          print("Iteration {}".format(i))
        eta = 0.9**i * eta0
        pi = frepsa_step(env, pi, (f, fp), eta)
        pi_all.append(pi)
      pi_alpha.append(pi_all)
    pi_alpha_all.append(pi_alpha)
  return pi_alpha_all


# ==== Data saving and loading
def save(outfile, meta, ret_alpha_all):
  # Metadata
  with open(outfile + '.json', 'w') as f:
    json.dump(meta, f)
  # Return trajectories
  np.save(outfile + '.npy', ret_alpha_all)


def load(infile):
  # Metadata
  with open(infile + '.json', 'r') as f:
    meta = json.load(f)
  # Return trajectories
  ret_alpha_all = np.load(infile + '.npy')
  return meta, ret_alpha_all


# ==== Plotting
def plot_returns(ax, alpha_all, ret_alpha_all):
  for alpha, ret_all in zip(alpha_all, ret_alpha_all):
    ret_mean = np.mean(ret_all, axis=0)
    ret_std = np.std(ret_all, axis=0)
    ax.errorbar(np.arange(ret_mean.size), ret_mean, 3*ret_std,
                label='alpha = {}'.format(alpha))


# ================================== fREPS ================================== #
def add_to_data(data, pi, th, lam, kap):
  data['pi'].append(pi)
  data['th'].append(th)
  data['lam'].append(lam)
  data['kap'].append(kap)


def freps_dual(D, eta, phi, f_div):
  """Create dual objective for fREPS."""
  A, DPhi, sa_multiplicities, D_sorted_keys = adv(D, phi)
  _, _, fcp, fc = f_div
  def dual(th, lam, kap):
    y = (A(th) - lam + kap) / eta
    w = fcp(y)
    g = eta * np.average(fc(y), weights=sa_multiplicities) + lam
    dg_dth = np.average(np.diag(w) @ DPhi, axis=0, weights=sa_multiplicities)
    dg_dlam = -np.average(w, weights=sa_multiplicities) + 1.0
    dg_dkap = np.average(np.diag(w), axis=0, weights=sa_multiplicities)
    return g, np.r_[dg_dth, dg_dlam, dg_dkap]
  return dual, A, DPhi, DPhi.shape, D_sorted_keys


def dual_vars(x, n_th):
  th = x[:n_th]
  lam = x[n_th]
  kap = x[n_th+1:]
  return th, lam, kap


def dual_x0(A, alpha, eta, n_th, rnd):
  th0 = rnd.randn(n_th)
  A0 = A(th0)
  if alpha > 1:
    lam0 = np.min(A0) + eta / (alpha - 1) - 1
  elif alpha < 1:
    lam0 = np.max(A0) - eta / (1 - alpha) + 1
  else:
    lam0 = np.mean(A0)
  kap0 = np.zeros(A0.size)
  return np.r_[th0, lam0, kap0]


def dual_arg(A, eta, n_th, x):
  th, lam, kap = dual_vars(x, n_th)
  y = (A(th) - lam + kap) / eta
  return y


def dual_arg_jac(dA, eta, n_th, x):
  th, lam, kap = dual_vars(x, n_th)
  d_dth = dA
  d_dlam = -np.ones((d_dth.shape[0], 1))
  d_dkap = np.eye(kap.size)
  return np.hstack((d_dth, d_dlam, d_dkap)) / eta


dual_opts = dict(
  jac=True,
  method='slsqp',
  options=dict(
    disp=False,
    iprint=2,
    maxiter=1000,
    ftol=1e-6
  )
)
def freps_pe(dual, A, dA, n_th, n_kap, rnd, alpha, eta):
  # Objective
  g = lambda x: dual(*dual_vars(x, n_th))
  # Bounds
  bnds = np.ones((n_th+1+n_kap, 1)) * np.array([-1e6, 1e6])
  # kap >= 0.0
  bnds[n_th+1:, 0] = 0.0
  # Constraints
  eps = 1e-3  # slack in the constraint
  y = partial(dual_arg, A, eta, n_th)
  dy = partial(dual_arg_jac, dA, eta, n_th)
  cons = (
    {  # 1 + (alpha - 1) * y - eps > 0
      'type': 'ineq',
      'fun': lambda x: 1 + (alpha - 1) * y(x) - eps,
      'jac': lambda x: (alpha - 1) * dy(x)
     },
  )
  # Solve
  success = False
  while not success:
    x0 = dual_x0(A, alpha, eta, n_th, rnd)
    sol = sp.optimize.minimize(g, x0, bounds=bnds, constraints=cons,
                               **dual_opts)
    success = sol.success
    if not success:
      print("Optimization failed!")
      print(sol)
  return dual_vars(sol.x, n_th)


def freps_pi(env, pi, f_div, A, sa_keys, eta, th, lam, kap):
  _, _, fcp, _ = f_div

  # Set advantage close to average for all (s,a) pairs
  A_SA = lam * np.ones((env.nS, env.nA))
  # For visited (s,a) pairs, use computed advantage
  A_SA[tuple(zip(*sa_keys))] = A(th)

  # Set kappa to zero for all (s,a) pairs
  Kap = np.zeros((env.nS, env.nA))
  # For visited (s,a) pairs, use optimal kappa
  Kap[tuple(zip(*sa_keys))] = kap

  # Update policy
  pi_new = np.copy(pi)
  pi_new *= fcp((A_SA - lam + Kap) / eta)
  pi_new /= np.sum(pi_new, axis=1, keepdims=True)
  return pi_new


def sim_freps(env, etaf, phi, seeds, alphas, n_sim, n_iter, n_steps, **kwargs):
  """Gather `n_sim` learning trajectories per `alpha`, each `n_iter` long."""
  A_space = np.arange(env.nA)
  sim_orchestra = {}
  for alpha in alphas:
    print("Alpha = {}".format(alpha))
    # Seed to start from the same state for every alpha
    env.seed(seeds[0])
    rnd = np.random.RandomState(seeds[1])
    # Fix alpha during policy iteration
    f_div = Fa(alpha)
    sim_ensemble = []
    for k in range(n_sim):
      print("Simulation {}".format(k))
      # Start every simulation from a random policy
      pi = np.ones((env.nS, env.nA)) / env.nA
      sim_traj = {'pi':[pi], 'th':[], 'lam':[], 'kap':[]}
      t0 = time.time()
      for i in range(n_iter):
        # Gather on-policy data
        D = gather_data(env, lambda s: rnd.choice(A_space, p=pi[s]), n_steps)
        # Critic
        eta = etaf(i)
        if alpha == 1.0:
          dual, A, n_th, sa_keys = reps_dual(D, eta, phi)
          th, lam = reps_pe(dual, n_th, rnd)
          kap = 0
        else:
          dual, A, dA, (n_kap, n_th), sa_keys = freps_dual(D, eta, phi, f_div)
          th, lam, kap = freps_pe(dual, A, dA, n_th, n_kap, rnd, alpha, eta)
        # Actor
        if alpha == 1.0:
          pi_new = reps_pi(env, pi, A, sa_keys, eta, th, lam)
        else:
          pi_new = freps_pi(env, pi, f_div, A, sa_keys, eta, th, lam, kap)
        # Save iteration snapshot
        add_to_data(sim_traj, pi, th, lam, kap)
        if (i+1) % 10 == 0:
          t1 = time.time()
          print("Iteration {}: lamda = {}, time = {}".format(i+1, lam, t1-t0))
          t0 = t1
        # Next step
        pi = pi_new
      sim_ensemble.append(sim_traj)
    sim_orchestra[alpha] = sim_ensemble
  return sim_orchestra


# =============================== fREPS Light =============================== #
def f_dual(D, eta, phi, f_div):
  A, DPhi, sa_multiplicities, D_sorted_keys = adv(D, phi)
  _, _, fcp, fc = f_div
  def dual(th, lam):
    y = (A(th) - lam) / eta
    w = fcp(y)
    g = eta * np.average(fc(y), weights=sa_multiplicities) + lam
    dg_dth = np.average(np.diag(w) @ DPhi, axis=0, weights=sa_multiplicities)
    dg_dlam = -np.average(w, weights=sa_multiplicities) + 1.0
    return g, np.r_[dg_dth, dg_dlam]
  return dual, A, DPhi, D_sorted_keys


def f_dual_x0(A, alpha, eta, n_th, rnd):
  th0 = rnd.randn(n_th)
  A0 = A(th0)
  if alpha > 1:
    lam0 = np.min(A0) + eta / (alpha - 1) - 1
  elif alpha < 1:
    lam0 = np.max(A0) - eta / (1 - alpha) + 1
  else:
    lam0 = np.mean(A0)
  return np.r_[th0, lam0]


def f_pe(dual, A, dA, rnd, alpha, eta):
  # Objective
  g = lambda x: dual(x[:-1], x[-1])
  # Constraints
  eps = 1e-3  # slack in the constraint
  y = lambda x: (A(x[:-1]) - x[-1]) / eta
  dcons = (alpha - 1) * np.hstack((dA, -np.ones((dA.shape[0], 1)))) / eta
  cons = (
    {  # 1 + (alpha - 1) * y - eps > 0
      'type': 'ineq',
      'fun': lambda x: 1 + (alpha - 1) * y(x) - eps,
      'jac': lambda x: dcons
    },
  )
  # Solve
  success = False
  while not success:
    x0 = f_dual_x0(A, alpha, eta, dA.shape[1], rnd)
    sol = sp.optimize.minimize(g, x0, constraints=cons, **dual_opts)
    success = sol.success
    if not success:
      print("Optimization failed!")
      print(sol)
  return sol.x[:-1], sol.x[-1]


def f_pi(env, pi, f_div, A, sa_keys, eta, th, lam):
  _, _, fcp, _ = f_div

  # Set advantage close to average for all (s,a) pairs
  A_SA = lam * np.ones((env.nS, env.nA))
  # For visited (s,a) pairs, use computed advantage
  A_SA[tuple(zip(*sa_keys))] = A(th)
  y = fcp((A_SA - lam) / eta)
  # print(np.min(y))
  # Update policy
  pi_new = np.copy(pi)
  pi_new *= fcp((A_SA - lam) / eta)
  pi_new /= np.sum(pi_new, axis=1, keepdims=True)
  return pi_new


def sim_f(env, etaf, phi, seeds, alphas, n_sim, n_iter, n_steps, **kwargs):
  """Gather `n_sim` learning trajectories per `alpha`, each `n_iter` long."""
  A_space = np.arange(env.nA)
  sim_orchestra = {}
  for alpha in alphas:
    print("Alpha = {}".format(alpha))
    # Seed to start from the same state for every alpha
    env.seed(seeds[0])
    rnd = np.random.RandomState(seeds[1])
    # Fix alpha during policy iteration
    f_div = Fa(alpha)
    sim_ensemble = []
    for k in range(n_sim):
      print("Simulation {}".format(k))
      # Start every simulation from a random policy
      pi = np.ones((env.nS, env.nA)) / env.nA
      sim_traj = {'pi':[pi], 'th':[], 'lam':[], 'kap':[]}
      t0 = time.time()
      for i in range(n_iter):
        # Gather on-policy data
        D = gather_data(env, lambda s: rnd.choice(A_space, p=pi[s]), n_steps)
        # Critic
        eta = etaf(i)
        if alpha == 1.0:
          dual, A, n_th, sa_keys = reps_dual(D, eta, phi)
          th, lam = reps_pe(dual, n_th, rnd)
        else:
          dual, A, dA, sa_keys = f_dual(D, eta, phi, f_div)
          th, lam = f_pe(dual, A, dA, rnd, alpha, eta)
        # Actor
        if alpha == 1.0:
          pi_new = reps_pi(env, pi, A, sa_keys, eta, th, lam)
        else:
          pi_new = f_pi(env, pi, f_div, A, sa_keys, eta, th, lam)
        # Save iteration snapshot
        add_to_data(sim_traj, pi, th, lam, None)
        if (i+1) % 10 == 0:
          t1 = time.time()
          print("Iteration {}: lamda = {}, time = {}".format(i+1, lam, t1-t0))
          t0 = t1
        # Next step
        pi = pi_new
        # print(pi)
      sim_ensemble.append(sim_traj)
    sim_orchestra[alpha] = sim_ensemble
  return sim_orchestra
