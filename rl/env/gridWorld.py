import numpy as np
import matplotlib.pyplot as plt
import os
from gym.envs.toy_text import discrete
from matplotlib.offsetbox import ( OffsetImage, AnnotationBbox)
from matplotlib.collections import LineCollection

env_path = os.path.abspath(os.path.dirname(__file__))
img_path = os.path.join(env_path, "images")

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
STAY = 4

class GridWorldEnv(discrete.DiscreteEnv):
    """
    A gridworld envrionment used for the implementation of reinforcement learning algorithms. 
    In this setup, the gridworld is a simulation of a flat or apartment of 9 x 10 states in total. 
    An agent or a robot tries to clean the room by collecting dirt.
    States:
        "O" denotes extremely dangerous states that the robot must avoid. Reward (O) = -1e5
        "D" denotes Dirt to be collected by the robot. Reward (D) = 35
        "W" denotes Water that robot should try to avoid. Reward (W) = -100
        "C" denotes Cat which may badly damage the robot. Reward (C) = -3000
        "T" denotes Toy that the robot enjoys playing with them. Reward (T) = 1000
    
    Actions:
        Down
        Right
        Up
        Left
        Stay
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape=[9, 10]):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')
        self.gridword = self.genGridWorld()

        self.shape = shape

        nS = np.prod(shape)
        nA = 5

        # self.state = np.random.choice(range(nS))

        MAX_Y = shape[0]
        MAX_X = shape[1]

        P = {}
        grid = np.arange(nS).reshape(shape)
        # Efficient multi-dimensional iterator object to iterate over arrays.
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            P[s] = {a: [] for a in range(nA)}

            is_done = lambda s: s == 46

            reawrd = 0 if is_done(s) else -1

            ns_up = s if y  ==  0 else s - MAX_X
            ns_right = s if x == (MAX_X -1 ) else s + 1
            ns_down = s if y == (MAX_Y - 1) else s + MAX_X
            ns_left = s if x == 0 else s -1
            ns_stay = s

            y_up, x_up = divmod(ns_up, MAX_X)
            y_right, x_right = divmod(ns_right, MAX_X)
            y_down, x_down = divmod(ns_down, MAX_X)
            y_left, x_left = divmod(ns_left, MAX_X)
            y_stay, x_stay = divmod(ns_stay, MAX_X)
            # # (prob, next_state, reward, done)
            P[s][UP]    = [(1.0, ns_up,     self.gridword[y_up, x_up],       is_done(ns_up))]
            P[s][RIGHT] = [(1.0, ns_right,  self.gridword[y_right, x_right], is_done(ns_right))]
            P[s][DOWN]  = [(1.0, ns_down,   self.gridword[y_down, x_down],   is_done(ns_down))]
            P[s][LEFT]  = [(1.0, ns_left,   self.gridword[y_left, x_left],   is_done(ns_left))]
            P[s][STAY]  = [(1.0, ns_stay,   self.gridword[y_stay, x_stay],   is_done(ns_stay))]

            it.iternext()

        # Initial state distribution is uniform
        isd = np.ones(nS) / nS

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        super(GridWorldEnv, self).__init__(nS, nA, P, isd)

    def genGridWorld(self):
        """
        Generate a grid world environment
        :return: grid_world: Numpy array representing the grid world
        """
        O = -1e5  # Dangerous places to avoid
        D = 35  # Dirt
        W = -100  # Water
        C = -3000  # Cat
        T = 1000  # Toy
        # grid_list = {0: '', O: 'O', D: 'D', W: 'W', C: 'C', T: 'T'}
        grid_world = np.array([[0, O, O, 0, 0, O, O, 0, 0, 0],
                               [0, 0, 0, 0, D, O, 0, 0, D, 0],
                               [0, D, 0, 0, 0, O, 0, 0, O, 0],
                               [O, O, O, O, 0, O, 0, O, O, O],
                               [D, 0, 0, D, 0, O, T, D, 0, 0],
                               [0, O, D, D, 0, O, W, 0, 0, 0],
                               [W, O, 0, O, 0, O, D, O, O, 0],
                               [W, 0, 0, O, D, 0, 0, O, D, 0],
                               [0, 0, 0, D, C, O, 0, 0, D, 0]])
        return grid_world

    def showWorld(self, grid_world, tlt, figure_size=(10,8), value_plot=False, show=False, savefig=None):
        """
        
        :param grid_world: 
        :param tlt: 
        :param figure_size: 
        :return: 
        """
        fig = plt.figure(figsize=figure_size, dpi=80)
        ax = fig.add_subplot(1,1,1)
        ax.set_title(tlt,weight='bold', fontsize='x-large' )
        ax.set_xticks(np.arange(0.5,10.5,1))
        ax.set_yticks(np.arange(0.5,9.5,1))
        ax.grid(color='b', linestyle='-', linewidth=0.5)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.imshow(grid_world, interpolation='nearest', cmap='copper')
        if value_plot == True:
            for x in range(grid_world.shape[0]):
                for y in range(grid_world.shape[1]):
                    ax.annotate("{0:.01f}".format(grid_world[x,y]), xy=(y,x),horizontalalignment='center',color='r')
        if show == True:
            plt.show()
        if savefig is not None:
            plt.savefig(savefig)
        return ax, fig

    def movingAgent(self,grid_world, ax, pre_state, pre_states):
        def showimage(ax, image, zoom_rate):
            arr_img = plt.imread(image)
            imagebox = OffsetImage(arr_img, zoom=zoom_rate)
            imagebox.image.axes = ax
            ab = AnnotationBbox(imagebox, [y, x],
                                xybox=(0, 0),
                                xycoords='data',
                                boxcoords="offset points",
                                )
            ax.add_artist(ab)
        patch = plt.Circle((5, -5), 0.4, fc='g')
        for x in range(grid_world.shape[0]):
            for y in range(grid_world.shape[1]):
                state = x * grid_world.shape[1] + y
                if state == pre_state:
                    patch.center = (y, x)
                    ax.add_patch(patch)
                if grid_world[x,y] == 35: # dirt
                    showimage(ax=ax, image=img_path + "/dirt.png", zoom_rate=0.035)
                elif grid_world[x,y] == -100:
                    showimage(ax=ax, image=img_path + "/water.png",zoom_rate=0.05)
                elif grid_world[x,y] == -3000:
                    showimage(ax=ax, image=img_path + "/cat.png",zoom_rate=0.035)
                elif grid_world[x,y] == 1000:
                    showimage(ax=ax, image=img_path + "/gold.jpeg",zoom_rate=0.075)
        if len(pre_states) > 1:
            (x_pre, y_pre) = divmod(np.array(pre_states), grid_world.shape[1])
            line_x = np.linspace(0, 4*np.pi, 20)
            line_width = 1 + line_x[:-1]
            points = np.array([y_pre, x_pre]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, linewidths=line_width[:len(x_pre)-1], linestyles=':', cmap='cool')
            ax.add_collection(lc)


    def showImageState(self, grid_world, ax, show=False):
        def showimage(ax, image, zoom_rate):
            arr_img = plt.imread(image)
            imagebox = OffsetImage(arr_img, zoom=zoom_rate)
            imagebox.image.axes = ax
            ab = AnnotationBbox(imagebox, [y, x],
                                xybox=(0, 0),
                                xycoords='data',
                                boxcoords="offset points",
                                )
            ax.add_artist(ab)
        for x in range(grid_world.shape[0]):
            for y in range(grid_world.shape[1]):
                # state = x * grid_world.shape[1] + y
                # ax.annotate(x*10+y, xy=(y,x),horizontalalignment='right',color='r')
                if grid_world[x,y] == 35: # dirt
                    showimage(ax=ax, image=img_path + "/dirt.png", zoom_rate=0.07)
                elif grid_world[x,y] == -100:
                    showimage(ax=ax, image=img_path + "/water.png",zoom_rate=0.11)
                elif grid_world[x,y] == -3000:
                    showimage(ax=ax, image=img_path + "/cat.png",zoom_rate=0.07)
                elif grid_world[x,y] == 1000:
                    showimage(ax=ax, image=img_path + "/gold.jpeg",zoom_rate=0.15)
        if show == True:
            plt.show()

    def showPolicy(self, policy, ax, show=False, savefig=None):
        for x in range(policy.shape[0]):
            for y in range(policy.shape[1]):
                action_count = 0
                for action, action_prob in enumerate(policy[x,y,:]):
                    if action_prob != 0:
                        action_count += 1
                        if action == 0:
                            ax.annotate(r'$\uparrow$', xy=(y,x), va ='bottom', ha='center', color='r', weight='bold', fontsize='x-large')
                        elif action == 1:
                            ax.annotate(r'$\rightarrow$', xy=(y,x), va='center', ha='left', color='r', weight='bold', fontsize='x-large')
                        elif action == 2:
                            ax.annotate(r'$\downarrow$', xy=(y,x), va ='top', ha='center',color='r', weight='bold', fontsize='x-large')
                        elif action == 3:
                            ax.annotate(r'$\leftarrow$', xy=(y,x), va='center', ha='right', color='r', weight='bold', fontsize='x-large')
                        elif action == 4:
                            if action_count == 1:
                                ax.annotate(r'$\perp$', xy=(y,x), va='center', ha='center', color='r', weight='bold', fontsize=20)
        if show == True:
            plt.show()

        if savefig is not None:
            plt.savefig(savefig)

