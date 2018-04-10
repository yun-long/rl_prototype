import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from functools import partial
# plt.style.use("ggplot")
#
Nfeval = 1
x_tmp, y_tmp = [], []
grad_x_tmp, grad_y_tmp = [], []

def non_convex_fn(x):
    return x ** 2 + 15 * np.sin(x)

def prime_f(grad, x0, y0):
    return lambda x: grad * (x - x0) + y0

def callable_f(x):
    global Nfeval, x_tmp, y_tmp, grad_x_tmp, grad_y_tmp
    eps = np.sqrt(np.finfo(float).eps)
    x_tmp.append(x[0])
    y_tmp.append(non_convex_fn(x[0]))
    x0 = np.array([x[0]])
    gradx = optimize.approx_fprime(x0, non_convex_fn, eps)
    y0 = non_convex_fn(x0)
    fprime = prime_f(gradx, x0, y0)
    x = np.linspace(x0-1, x0+1, 100)
    grad_f = fprime(x)
    grad_x_tmp.append(x)
    grad_y_tmp.append(grad_f)
    print("{0:4d} {1:1.4f} {2:2.4f}".format(Nfeval, x[0], non_convex_fn(x[0])))
    Nfeval += 1

def gradient_descend(f, x0, alpha):
    eps = np.sqrt(np.finfo(float).eps)
    x, y = [], []
    x.append(x0)
    y.append(f(x0))
    for i in range(50):
        gradx0 = optimize.approx_fprime(x0, f, eps)
        x0 = x0 - alpha * gradx0
        x.append(x0)
        y.append(f(x0))

    return x, y

if __name__ == '__main__':
    x = np.linspace(-10, 13, 1000)
    y = non_convex_fn(x)
    eps = np.sqrt(np.finfo(float).eps)
    x0 = np.array([12])
    gradx0 = optimize.approx_fprime(x0, non_convex_fn, eps)
    y0 = non_convex_fn(x0)
    fprime = prime_f(gradx0, x0, y0)
    xtmp = np.linspace(x0-1, x0+1, 100)
    gradtmp = fprime(xtmp)
    fig, axes = plt.subplots(2, 2)
    result = optimize.minimize(non_convex_fn, x0, method='BFGS', tol=1e-6, callback=callable_f, options={'disp': True})
    # ax.plot(xtmp, gradtmp, 'r')
    # ax.plot(x0, y0, 'ro')
    alphas = [0.01, 0.02, 0.1, 0.5]
    for i, ax in enumerate(axes.flatten()):
        gd_xs, gd_ys = gradient_descend(non_convex_fn, x0, alphas[i])
        ax.plot(x, y)
        # ax.scatter(x_tmp, y_tmp, color='r')
        # ax.plot(result.x, non_convex_fn(result.x), 'ro')
        # for i, gradx in enumerate(grad_x_tmp):
            # grady = grad_y_tmp[i]
            # ax.plot(gradx, grady, color='r')
        ax.plot(gd_xs, gd_ys)
        ax.axis('off')
        for side in ['bottom', 'right', 'top', 'left']:
            ax.spines[side].set_visible(False)
        ax.set_title(r"$\alpha={0:.2f}$".format(alphas[i]))
    plt.show()

