import numpy as np
import sys
from gym.envs.toy_text import discrete
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class CliffWalkingEnv(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi']}

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta):
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        reward = -100.0 if self._cliff[tuple(new_position)] else -1.0
        is_done = self._cliff[tuple(new_position)] or (tuple(new_position) == (3,11))
        return [(1.0, new_state, reward, is_done)]

    def __init__(self):
        self.shape = (4, 12)
        self.cliff_env = self.genCliff()
        nS = np.prod(self.shape)
        nA = 4

        # Cliff Location
        self._cliff = np.zeros(self.shape, dtype=np.bool)
        self._cliff[3, 1:-1] = True

        # Calculate transition probabilities
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = { a : [] for a in range(nA) }
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

        # We always start in state (3, 0)
        isd = np.zeros(nS)
        isd[np.ravel_multi_index((3,0), self.shape)] = 1.0

        super(CliffWalkingEnv, self).__init__(nS, nA, P, isd)

    def genCliff(self):
        """
        Generate a grid world environment
        :return: grid_world: Numpy array representing the grid world
        """
        S = 700
        G = 1000
        C = 0
        X = 500
        # grid_list = {0: '', O: 'O', D: 'D', W: 'W', C: 'C', T: 'T'}
        grid_world = np.array([[X, X, X, X, X, X, X, X, X, X, X, X],
                               [X, X, X, X, X, X, X, X, X, X, X, X],
                               [X, X, X, X, X, X, X, X, X, X, X, X],
                               [S, C, C, C, C, C, C, C, C, C, C, G]])
        return grid_world

    def showWorld(self, tlt, figure_size=(8, 3),  show=False):
        """

        :param grid_world: 
        :param tlt: 
        :param figure_size: 
        :return: 
        """
        fig = plt.figure(figsize=figure_size, dpi=80)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(tlt, weight='bold', fontsize='x-large')
        ax.set_xticks(np.arange(0.5, 10.5, 1))
        ax.set_yticks(np.arange(0.5, 9.5, 1))
        ax.grid(color='b', linestyle='-', linewidth=0.5)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.imshow(self.cliff_env, interpolation='nearest', cmap='seismic')
        if show == True:
            plt.show()
        return ax, fig

    def showImageState(self, ax, show=False):
        for x in range(self.cliff_env.shape[0]):
            for y in range(self.cliff_env.shape[1]):
                if self.cliff_env[x,y] == 700: # dirt
                    ax.annotate('S', xy=(y, x), va='center', ha='center', color='black', weight='bold',
                                fontsize='x-large')
                elif self.cliff_env[x,y] == 1000:
                    ax.annotate('G', xy=(y, x), va='center', ha='center', color='black', weight='bold',
                                fontsize='x-large')
                if x == 3 and y == 5:
                    ax.annotate('Cliff', xy=(y, x), va='center', ha='center', color='white',weight='bold')
        if show == True:
            plt.show()


    def showPolicy(self, policy, ax, show=False):
        for x in range(policy.shape[0]):
            for y in range(policy.shape[1]):
                for action, action_prob in enumerate(policy[x,y,:]):
                    if action_prob != 0:
                        if action == UP:
                            ax.annotate(r'$\uparrow$', xy=(y,x), va ='bottom', ha='center', color='r', weight='bold', fontsize='x-large')
                        elif action == RIGHT:
                            ax.annotate(r'$\rightarrow$', xy=(y,x), va='center', ha='left', color='r', weight='bold', fontsize='x-large')
                        elif action == DOWN:
                            ax.annotate(r'$\downarrow$', xy=(y,x), va ='top', ha='center',color='r', weight='bold', fontsize='x-large')
                        elif action == LEFT:
                            ax.annotate(r'$\leftarrow$', xy=(y,x), va='center', ha='right', color='r', weight='bold', fontsize='x-large')
        if show == True:
            plt.show()

    def movingAgent(self, ax, pre_state, pre_states, action):
        patch = plt.Circle((5, -5), 0.4, fc='brown')
        for x in range(self.cliff_env.shape[0]):
            for y in range(self.cliff_env.shape[1]):
                state = x * self.cliff_env.shape[1] + y
                if state == pre_state:
                    if action == UP:
                        ax.annotate(r'$\uparrow$', xy=(y, x), va='bottom', ha='center', color='r', weight='bold',
                                    fontsize=30)
                    elif action == RIGHT:
                        ax.annotate(r'$\rightarrow$', xy=(y, x), va='center', ha='left', color='r', weight='bold',
                                    fontsize=30)
                    elif action == DOWN:
                        ax.annotate(r'$\downarrow$', xy=(y, x), va='top', ha='center', color='r', weight='bold',
                                    fontsize=30)
                    elif action == LEFT:
                        ax.annotate(r'$\leftarrow$', xy=(y, x), va='center', ha='right', color='r', weight='bold',
                                    fontsize=30)
                    patch.center = (y, x)
                    ax.add_patch(patch)
                if self.cliff_env[x,y] == 900: # dirt
                    ax.annotate('S', xy=(y, x), va='center', ha='center', color='black', weight='bold',
                                fontsize='x-large')
                elif self.cliff_env[x,y] == 1:
                    ax.annotate(r'$\uparrow$', xy=(y, x), va='center', ha='center', color='black')
                elif self.cliff_env[x,y] == 2:
                    ax.annotate(r'$\uparrow$$\uparrow$', xy=(y, x), va='center', ha='center', color='black', weight='bold',
                                fontsize='x-large')
                elif self.cliff_env[x,y] == 1000:
                    ax.annotate('G', xy=(y, x), va='center', ha='center', color='black', weight='bold',
                                fontsize='x-large')
        if len(pre_states) > 1:
            (x_pre, y_pre) = divmod(np.array(pre_states), self.cliff_env.shape[1])
            line_x = np.linspace(0, 4*np.pi, 100)
            line_width = 1 + line_x[:-1]
            points = np.array([y_pre, x_pre]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, linewidths=line_width[:len(x_pre)-1], linestyles=':', colors='brown')
            ax.add_collection(lc)