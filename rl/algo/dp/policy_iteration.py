#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import imageio
# from gym.envs.toy_text.cliffwalking import CliffWalkingEnv
from rl.env.gridWorld import GridWorldEnv
from rl.misc.utilies import ROOT_PATH, get_dirs
from rl.misc.utilies import fig_to_image


RESULT_PATH = get_dirs(os.path.join(ROOT_PATH, "results"))


class Agent:

    def __init__(self, env):
        self.V = np.zeros(env.nS)
        self.policy = np.ones([env.nS, env.nA]) / env.nA
        self.init_state = 0
        self.state = None
        self.terminate_state = 46

    def policy_evaluation(self, env, policy, theta=0.00001, discount_factor=0.9):
        V = np.zeros(env.nS)
        while True:
            Delta = 0.0
            for s in range(env.nS):
                v = 0
                for a, action_prob in enumerate(policy[s]):
                    for prob, next_state, reward, done in env.P[s][a]:
                        v += action_prob * prob * (reward + discount_factor * V[next_state])
                Delta = max(Delta, np.abs(v-V[s]))
                V[s] = v
            if Delta < theta:
                return V

    def policy_improvement(self, env, V, policy, discont_factor=0.9):
        converged = True
        for s in range(env.nS):
            action = np.argmax(policy[s])
            action_value = np.zeros(env.nA)
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_value[a] += prob * (reward + discont_factor * V[next_state])
            best_action = np.argmax(action_value)
            if action != best_action:
                converged = False
            policy[s] = np.eye(env.nA)[best_action]
        return policy, converged

    def policy_iteration(self, env):
        # Initialization
        V = self.V
        policy = self.policy
        # policy evaluation and policy imporvement
        count = 0
        while True:
            count += 1
            print("Iteration {}".format(count))
            # update value function
            V = self.policy_evaluation(env, policy)
            # update policy
            policy, converged = self.policy_improvement(env, V, policy)
            #
            if converged == True:
                break
        self.V = V
        self.policy = policy

    def showGridworld(self, env, show=False):
        env_gridworld = env.genGridWorld()
        ax, _ = env.showWorld(env_gridworld, tlt="GridWorld Environment")
        env.showImageState(env_gridworld, ax, show)

    def showValueFun(self, env, show=False):
        V_show = self.V.reshape((9, 10))
        savefig = os.path.join(RESULT_PATH, "DP/pi_value.png")
        ax, _ = env.showWorld(V_show, tlt="GridWorld Environment", value_plot=True, show=show, savefig=savefig)

    def showPolicy(self, env, show=False):
        env_gridworld = env.genGridWorld()
        policy_show = self.policy.reshape([9, 10, 5])
        # show policy
        grid_world = os.path.join(RESULT_PATH, "DP/gridworld.png")
        policy = os.path.join(RESULT_PATH, "DP/pi_policy.png")
        ax, _ = env.showWorld(grid_world=env_gridworld, tlt="GridWorld Environment", savefig=None)
        env.showPolicy(policy_show, ax, show=show,savefig=policy)

    def showExamples(self, env):
        init_state = np.array([0,29,82])
        gridworld = env.genGridWorld()
        cout = 0
        frames = []
        for i in range(3):
            step = 0
            pre_states = []
            self.state = init_state[i]
            converge = False
            while True:
                pre_states.append(self.state)
                action = np.argmax(self.policy[self.state])
                _, next_state, _, _ = env.P[self.state][action][0]
                ax, fig = env.showWorld(gridworld, tlt="Round {0}, Step {1}".format(i+1, step))
                env.movingAgent(gridworld, ax, self.state, pre_states)
                image = fig_to_image(fig)
                frames.append(image)
                plt.close()
                self.state = next_state
                if converge == True:
                    cout +=1
                    break
                if self.state == self.terminate_state:
                    converge = True
                step += 1
                cout += 1
        file_dir = get_dirs(os.path.join(RESULT_PATH, "DP"))
        imageio.mimsave(os.path.join(file_dir, "pi_test.gif"), frames, fps=5)

def main():
    # define a gridworld environment
    env = GridWorldEnv()
    # create an agent
    agent = Agent(env)
    #
    agent.policy_iteration(env)
    #
    # agent.showGridworld(env, show=False)
    #
    agent.showValueFun(env, show=False)
    #
    agent.showPolicy(env, show=False)
    #
    agent.showExamples(env)
    print("Results are saved in {}".format(RESULT_PATH))

if __name__ == '__main__':
    main()