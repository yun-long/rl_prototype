import time
import os
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas



def get_dirs(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def discount_norm_rewards(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)
    return  discounted_rewards

def stable_log_exp_sum(x):
    """
    y = np.log(np.sum(np.exp(x)) # not stable
      = np.max(x) + np.log(np.sum(np.exp(x - np.max(x))) # stable
    :param x:
    :return:
    """
    max_x = np.max(x)
    y = max_x + np.log(np.sum(np.exp(x-max_x)))
    return y

def fig_to_image(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    width = int(width)
    height = int(height)
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(height, width, 3)
    return  image

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def print_envinfo(env, disc_a=False, disc_s=False):
    # np.random.seed(1234)
    # env.seed(1234)
    print("Action space : ", env.action_space)
    if not disc_a:
        print("Action space low : ", env.action_space.low)
        print("Action space high: ", env.action_space.high)
    print("Observation space : ", env.observation_space)
    if not disc_s:
        print("Observation space low : ", env.observation_space.low)
        print("Observation space high: ", env.observation_space.high)


def opt_policy_demo(env, policy):
    obs = env.reset()
    while True:
        action = policy(obs)
        next_obs, reward, done, _ = env.step(action)
        obs = next_obs
        env.render()
        if done:
            time.sleep(0.5)
            obs = env.reset()

