import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from rl.env.gridWorld import GridWorldEnv
from rl.misc.utilies import ROOT_PATH, get_dirs
#
RESULT_PATH = get_dirs(os.path.join(ROOT_PATH, "results"))
#
SAVE_PATH = get_dirs(os.path.join(RESULT_PATH, "DP"))

def value_iteration(env, theta=0.00001, disount_factor=0.9):
    """
    Value Iteration Algorithm
    :param env: 
    :param theta: 
    :param disount_factor: 
    :return: 
    """
    # one step look ahead
    def one_step_lookahead(V, s):
        action_value = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[s][a]:
                action_value[a] += prob * (reward + disount_factor * V[next_state])
        return action_value
    #
    V = np.zeros(env.nS)
    count_iteration = 0
    while True:
        count_iteration += 1
        Delta = 0.0
        for s in range(env.nS):
            action_value = one_step_lookahead(V, s)
            best_action_value = np.max(action_value)
            Delta = max(Delta, np.abs(best_action_value-V[s]))
            V[s] = best_action_value
        print("Iteration {0}".format(count_iteration))
        if Delta < theta:
            break
    # greedy policy
    def greedy_policy(env, V):
        policy = np.zeros([env.nS, env.nA])
        for s in range(env.nS):
            action_value = one_step_lookahead(V, s)
            best_action = np.argmax(action_value)
            policy[s, best_action] = 1
        return policy
    policy = greedy_policy(env, V)
    return V, policy

def main():
    # create a gridworld environment class
    gridworldEnv = GridWorldEnv()
    # show policy
    gridWord = gridworldEnv.genGridWorld()
    ax, _ = gridworldEnv.showWorld(grid_world=gridWord, tlt="GridWorld Environment")
    gridworldEnv.showImageState(gridWord, ax)
    # calculate value iteration and generate greedy policy
    V, policy = value_iteration(env=gridworldEnv)
    policy = policy.reshape((9,10,5))
    # show policy
    ax, _ = gridworldEnv.showWorld(grid_world=gridWord, tlt="GridWorld Environment")
    gridworldEnv.showPolicy(policy, ax, savefig=SAVE_PATH+"/vi_policy.png")
    print("Results are saved in {}".format(SAVE_PATH))
    #
    # plt.show()

if __name__ == '__main__':
    main()