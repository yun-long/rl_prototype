import numpy as np
from scipy.optimize import minimize
from functools import partial

# # ===================== REPS ===================
#
opts = dict(disp=False,
            iprint=2,
            maxiter=1000,
            ftol=1e-6)

# # ===================== REPS dual function version 0, for continous case ===================
def dual_fn_v0(rewards, features_diff, epsilon, N, inputs):
    """
    Dual function for continuous case, update both "v" and "eta".
    """
    param_eta = inputs[0]
    param_v = inputs[1:]
    adv = rewards + np.dot(features_diff, param_v)
    max_adv = np.max(adv)
    Z = np.exp((adv - np.max(adv))/ param_eta)
    g = param_eta * epsilon + max_adv + param_eta * np.log(np.sum(Z * 1. / N))
    grad_eta = epsilon + np.log(np.sum(Z * 1. / N)) - Z.dot(adv - max_adv) / (param_eta * np.sum(Z))
    grad_v = Z.dot(features_diff) / np.sum(Z)
    return g, np.hstack([grad_eta, grad_v])

def optimize_dual_fn(rewards, features_diff, init_eta, init_v, epsilon):
    N = rewards.shape[0]
    x0 = np.hstack([init_eta, init_v])
    bounds = [(-np.inf, np.inf) for _ in x0]
    bounds[0] = (1e-9, np.inf)
    #
    opt_fn = partial(dual_fn_v0, rewards, features_diff, epsilon, N)
    results = minimize(opt_fn, x0=x0, method="slsqp", jac=True, options=opts, bounds=bounds)
    if not results.success:
        print("Optimization failed!")
    eta = results.x[0]
    v = results.x[1:]
    adv = (rewards + np.dot(features_diff, v)) / eta
    weights = np.exp((adv - np.max(adv))) / np.sum(np.exp(adv - np.max(adv)))
    return eta, v, weights

# # ===================== REPS dual function version 1, for discrete case, update param_v only, light ===================
def dual_fn_v1(A, features_diff, eta, sa_n, param_v):
    """
    Dual function for distrete case, update parameter "v" only
    """
    adv = A(param_v)
    max_adv = np.max(adv)
    Z = np.exp((adv-max_adv) / eta)
    g = max_adv + eta * np.log(np.average(Z, weights=sa_n))
    dg = np.average(features_diff, axis=0, weights=Z*sa_n)
    return g, dg

def optimize_dual_fn_paramv(rewards, features_diff, init_eta, init_v, epsilon, sa_n):
    A = lambda v: rewards + np.dot(features_diff, v)
    x0 = init_v
    bounds = [(-np.inf, np.inf) for _ in x0]
    opt_fn = partial(dual_fn_v1, A, features_diff, init_eta, sa_n)
    results = minimize(opt_fn, x0, method='slsqp', jac=True, bounds=bounds, options=opts)
    if not results.success:
        print("Optimization falied!")
    v = results.x
    return v, A, results.fun

# # ===================== REPS dual function version 2, for discrete case, update param_v and param_eta ===================
def dual_fn_v2(A, features_diff, epsilon, sa_n, inputs):
    param_eta = inputs[0]
    param_v = inputs[1:]
    adv = A(param_v)
    max_adv = np.max(adv)
    Z = np.exp((adv-max_adv) / param_eta)
    g = param_eta * epsilon + max_adv + param_eta * np.log(np.average(Z, weights=sa_n))
    grad_eta = epsilon + np.log(np.average(Z, weights=sa_n)) - np.average((adv-max_adv),axis=0, weights=Z*sa_n) / param_eta
    grad_v = np.average(features_diff, axis=0, weights=Z*sa_n)
    return g, np.hstack([grad_eta, grad_v])

def optimize_dual_fn_params(rewards, features_diff, init_eta, init_v, epsilon, sa_n):
    A = lambda v : rewards + np.dot(features_diff,v)
    x0 = np.hstack([init_eta, init_v])
    bounds = [(-np.inf, np.inf) for _ in x0]
    bounds[0] = (1e-5, np.inf)
    opt_fn = partial(dual_fn_v2, A, features_diff, epsilon, sa_n)
    results = minimize(opt_fn,
                       x0=x0,
                       method="slsqp",
                       jac=True,
                       options=opts,
                       bounds=bounds)
    if not results.success:
        print("Optimization failed! \n")
    eta = results.x[0]
    v = results.x[1:]
    return eta, v, A, results.fun




# # ===================== fREPS  ===================

dual_opts = dict(
  jac=True,
  method='SLSQP',
  options=dict(
    disp=False,
    iprint=2,
    maxiter=1000,
    ftol=1e-6
  )
)

def alpha_fn(alpha=1.0):
    if alpha == 1.0:  # KL-divergence
        f = lambda x: x * np.log(x) - (x - 1)
        fp = lambda x: np.log(x)
        fcp = lambda y: np.exp(y)
        fc = lambda y: np.exp(y) - 1
    elif alpha == 0.0:  # Reverse KL
        f = lambda x: -np.log(x) + (x - 1)
        fp = lambda x: -1 / x + 1
        fcp = lambda y: 1 / (1 - y)
        fc = lambda y: -np.log(1 - y)
    else:
        f = lambda x: ((np.power(x, alpha) - 1) - alpha * (x - 1)) \
                      / (alpha * (alpha - 1))
        fp = lambda x: (np.power(x, alpha - 1) - 1) / (alpha - 1)
        fcp = lambda y: np.power(1 + (alpha - 1) * y, 1 / (alpha - 1))
        fc = lambda y: 1 / alpha * np.power(1 + (alpha - 1) * y, alpha / (alpha - 1)) \
                       - 1 / alpha
    return f, fp, fcp, fc

# # ===================== fREPS dual function version 0, for discrete case, light ===================

def alpha_fn_x0(alpha, A, eta, n_v, rnd):
    v0 = rnd.randn(n_v)
    A0 = A(v0)
    if alpha > 1.:
        lam0 = np.min(A0) + eta / (alpha - 1) - 1
    elif alpha < 1.:
        lam0 = np.max(A0) - eta / (1- alpha) + 1
    else:
        lam0 = np.mean(A0)
    return np.r_[v0, lam0]

def f_dual_v0(A, features_diff, sa_n, eta, alpha):
    _, _, fcp, fc = alpha_fn(alpha)
    def dual(inputs):
        param_v = inputs[:-1]
        param_lam = inputs[-1]
        y = (A(param_v) - param_lam) / eta
        w = fcp(y)
        g = eta * np.average(fc(y), weights=sa_n) + param_lam
        dg_v = np.average(np.diag(w) @ features_diff, axis=0, weights=sa_n)
        dg_lam = -np.average(w, weights=sa_n) + 1.0
        return g, np.r_[dg_v, dg_lam]
    return dual

def optimize_fdual_fn_v0(rewards, features_diff, sa_n, eta, alpha, rnd):
    n_v = features_diff.shape[1]
    A = lambda v : rewards + np.dot(features_diff, v)
    #
    opt_fn = f_dual_v0(A, features_diff, sa_n, eta, alpha)
    # Don't quite understand
    eps = 1e-3  # slack in the constraint
    y = lambda x: (A(x[:-1]) - x[-1]) / eta
    dcons = (alpha - 1) * np.hstack((features_diff, -np.ones((features_diff.shape[0], 1)))) / eta
    cons = (
        {  # 1 + (alpha - 1) * y - eps > 0
            'type': 'ineq',
            'fun': lambda x: 1 + (alpha - 1) * y(x) - eps,
            'jac': lambda x: dcons
        }
    )
    success = False
    while not success:
        x0 = alpha_fn_x0(alpha, A, eta, n_v, rnd)
        results = minimize(opt_fn, x0=x0, constraints=cons, **dual_opts)
        success = results.success
        if not success:
            print("Optimization failed! \n")
            print(results)
    #
    _, _, fcp, fc = alpha_fn(alpha)
    v = results.x[:-1]
    lamda = results.x[-1]
    return lamda, v, A, fcp

# # ===================== fREPS dual function version 1, for discrete case ===================

def alpha_fn_x0_v1(alpha, A, eta, n_v, rnd):
    v0 = rnd.randn(n_v)
    A0 = A(v0)
    if alpha > 1.:
        lam0 = np.min(A0) + eta / (alpha - 1) - 1
    elif alpha < 1.:
        lam0 = np.max(A0) - eta / (1- alpha) + 1
    else:
        lam0 = np.mean(A0)
    kap0 = np.zeros(A0.size)
    return np.r_[v0, lam0, kap0]

def dual_arg(A, eta, n_v, inputs):
    param_v = inputs[:n_v]
    param_lam = inputs[n_v]
    param_kappa = inputs[n_v+1:]
    y = (A(param_v) - param_lam + param_kappa) / eta
    return y

def dual_arg_jac(dA, eta, n_th, inputs):
    param_kappa = inputs[n_th+1:]
    d_dv = dA
    d_dlam = -np.ones((d_dv.shape[0], 1))
    d_dkap = np.eye(param_kappa.size)
    return np.hstack((d_dv, d_dlam, d_dkap)) / eta

def f_dual_v1(A, features_diff, sa_n, eta, alpha, n_v):
    _, _, fcp, fc = alpha_fn(alpha)
    def dual(inputs):
        param_v = inputs[:n_v]
        param_lam = inputs[n_v]
        param_kappa = inputs[n_v+1:]
        y = (A(param_v) - param_lam + param_kappa) / eta
        w = fcp(y)
        g = eta * np.average(fc(y), weights=sa_n) + param_lam
        dg_v = np.average(np.diag(w) @ features_diff, axis=0, weights=sa_n)
        dg_lam = -np.average(w, weights=sa_n) + 1.0
        # TODO: ....
        dg_kappa = np.average(np.diag(w), axis=0, weights=sa_n)
        return g, np.r_[dg_v, dg_lam, dg_kappa]
    return dual

def optimize_fdual_fn_v1(rewards, features_diff, sa_n, eta, alpha, rnd):
    n_kappa, n_v = features_diff.shape
    A = lambda v: rewards + np.dot(features_diff, v)
    opt_fn = f_dual_v1(A, features_diff, sa_n, eta, alpha,  n_v)
    #
    bounds = np.ones((n_v+1+n_kappa, 1)) * np.array([-1e6, 1e6])
    # kappa >= 0.0
    bounds[n_v+1:, 0] = 0.0
    #
    y = partial(dual_arg, A, eta, n_v)
    dy = partial(dual_arg_jac, features_diff, eta, n_v)
    eps = 1e-3
    cons = (
        {  # 1 + (alpha - 1) * y - eps > 0
            'type': 'ineq',
            'fun': lambda x: 1 + (alpha - 1) * y(x) - eps,
            'jac': lambda x: (alpha - 1) * dy(x)
        },
    )
    success = False
    while not success:
        x0 = alpha_fn_x0_v1(alpha, A, eta, n_v, rnd)
        results = minimize(opt_fn, x0=x0, bounds=bounds, constraints=cons, **dual_opts)
        success = results.success
        if not success:
            print("Optimization failed! \n")
            print(results)
    #
    _, _, fcp, fc = alpha_fn(alpha)
    v = results.x[:n_v]
    lamda = results.x[n_v]
    kappa = results.x[n_v+1:]
    return v, lamda, kappa, A, fcp


























