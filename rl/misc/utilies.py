import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def discount_norm_rewards(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    # discounted_rewards -= np.mean(discounted_rewards)
    # discounted_rewards /= np.std(discounted_rewards)
    return  discounted_rewards

def stable_log_exp_sum(x, N=None):
    """
    y = np.log(np.sum(np.exp(x)) / len(x)) # not stable
      = np.max(x) + np.log(np.sum(np.exp(x - np.max(x))) / len(x)) # stable
    :param x:
    :return:
    """
    max_x = np.max(x)
    if N is None:
        y = max_x + np.log(np.sum(np.exp(x-max_x)))
    else:
        y = max_x + np.log(np.sum(np.exp(x-max_x)) / N)
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