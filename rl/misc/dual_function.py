import numpy as np
from scipy.optimize import minimize
from functools import partial
#
opts = dict(disp=False, iprint=2, maxiter=1000, ftol=1e-6)

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

# # ===================== REPS dual function version 1, for discrete case ===================
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
    return init_eta, v, A, results.fun

# # ===================== REPS dual function version 2, for discrete case ===================
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



# # ===================== fREPS dual function version 0, for discrete case ===================
def fdual_fn_v1(A, features_diff, epsilon, sa_n, kappa, inputs):
    param_eta = inputs[0]
    param_v = inputs[1:]

    pass


def optimize_fdual_fn():
    pass

































