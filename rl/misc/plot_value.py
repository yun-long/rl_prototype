import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")


def plot_2D_value(env, value_fn, conti=False, show=True):
    fig = plt.figure()
    y_value = []
    x_state = []
    if conti: # if the state is continuous
        for state in np.linspace(env.observation_space.low[0], env.observation_space.high[0], 1000):
            x_state.append(state)
            y_value.append(value_fn.predict(state))
    else: # else the state is discrete
        for state in np.arange(env.observation_space.n):
            x_state.append(state)
            y_value.append(value_fn.predict(state))
    plt.plot(x_state, y_value)
    plt.xlabel("States")
    plt.ylabel("Values")
    plt.title("Value function")
    if show:
        plt.show()
    return fig

