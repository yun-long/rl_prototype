import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def plot_cost_mountain_car(env, approximator, step=0, num_tiles=20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num_tiles)
    #
    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(lambda _: -np.max(approximator.predict(_)), 2, np.dstack((X, Y)))
    #
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z,cmap=matplotlib.cm.coolwarm)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value')
    ax.set_zlim(0, 200)
    ax.set_title("Cost-to-go function, Episode: {0}".format(step))
    fig.colorbar(surf)
    return fig

def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))

