import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def plot_3D_multi_norm_dist_1(Mu, Sigma, show=True):
    """
    Plot Multivariate Normal Distribution.
    :param Mu: python list or numpy array, shape = (2,)
    :param Sigma: python list or numpy matrix, shape = (2, 2)
    :param show: if true, plot.show()
    :return: None
    """
    # Create grid and multivariate normal
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    rv = multivariate_normal(Mu, Sigma)
    # Make a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, rv.pdf(pos), cmap='viridis', linewidth=0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    if show == True:
        plt.show()

def plot_3D_multi_norm_dist_n(Mu_x, Mu_y, Sigma_x, Sigma_y, show=True):
    start = -1
    stop = 1
    length = stop - start

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for x_i, mu_x in enumerate(Mu_x):
        x = np.linspace(start + (x_i*length), stop + (x_i*length), 100)
        sigma_x = Sigma_x[x_i, x_i]
        mu_x += (x_i)*length
        for y_i, mu_y in enumerate(Mu_y):
            y = np.linspace(start + (y_i*length), stop + (y_i*length),100)
            sigma_y = Sigma_y[y_i, y_i]
            mu_y += (y_i)*length
            X, Y = np.meshgrid(x, y)
            pos = np.empty(X.shape + (2,))
            pos[:, :, 0] = X
            pos[:, :, 1] = Y
            rv = multivariate_normal([mu_x, mu_y], [[sigma_x, 0], [0, sigma_y]])
            ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
            ax.hold()

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.show()

def plot_1d_norm_dist_1(mu, sigma, show=True):
    x = np.linspace(mu-3*sigma, mu+3*sigma, 100)
    plt.figure()
    plt.plot(x, norm.pdf(x, mu, sigma))
    if show:
        plt.show()

def plot_1d_norm_dist_n(Mu, Sigma, show=True):
    fig = plt.figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.set_ylim(0,40)
    ax.set_xlim(-3,3)
    for i, mu in enumerate(Mu):
        sigma = Sigma[i, i]
        # length = (3*sigma)*2
        # mu += (i*length)
        x = np.linspace(mu-3*sigma, mu+3*sigma, 100)
        ax.plot(x, norm.pdf(x, mu, sigma))
    if show:
        plt.show()
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    width = int(width)
    height = int(height)
    info_dict = {"canvas": canvas, "height": height, "width":width}
    return info_dict

if __name__ == '__main__':
    num_features = 10
    Mu_w = np.zeros(num_features)
    Sigma_w = np.eye(num_features) * 1e-1
    plot_1d_norm_dist_1(Mu_w[0], Sigma_w[0, 0])
    plot_1d_norm_dist_n(Mu_w, Sigma_w)
    plot_3D_multi_norm_dist_1(Mu_w, Sigma_w)
    plot_3D_multi_norm_dist_n(Mu_w, Mu_w, Sigma_w,Sigma_w)

