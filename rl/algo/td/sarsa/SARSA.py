import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import imageio
from collections import defaultdict
from rl.misc.utilies import get_dirs, fig_to_image
from rl.env.windy_gridWorld import WindyGridworldEnv
# import constants_TD as C
#
ROOT_PATH = os.path.realpath("../../../../")
#
RESULT_PATH = get_dirs(os.path.join(ROOT_PATH, "results"))
#
SAVE_PATH = get_dirs(os.path.join(RESULT_PATH, "SARSA"))

class Agent_SARSA:

    def __init__(self, env):
        self.Q = defaultdict(lambda : np.zeros(env.nA))
        self.policy = np.ones([env.nS, env.nA]) / env.nA

    def epsilon_greedy_policy(self, Q, epsilon, nA):
        def policy_fn(observation):
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(Q[observation])
            A[best_action] += (1.0 - epsilon)
            return A
        return policy_fn

    def sarsa(self, env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
        policy = self.epsilon_greedy_policy(self.Q, epsilon, env.nA)
        for i_episode in range(num_episodes):
            state = env.reset()
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            for t in itertools.count():
                # take an action
                next_step, reward, done, _ = env.step(action)
                # pick next action
                next_action_probs = policy(next_step)
                next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
                # update Q
                TD_target = reward + discount_factor * self.Q[next_step][next_action]
                TD_delta = TD_target - self.Q[state][action]
                self.Q[state][action] += alpha * TD_delta
                if done:
                    break
                # update state and action
                state = next_step
                action = next_action
        self.policy = self.optimal_policy(env,self.Q)
        return self.Q, self.policy

    def optimal_policy(self, env, Q):
        for s in range(env.nS):
            best_action = np.argmax(Q[s])
            self.policy[s] = np.eye(env.nA)[best_action]
        return self.policy

    def saveExamples(self, env, title="Windy Grid World Environment", save_folder=None, epsilon=0.1):
        policy = self.epsilon_greedy_policy(self.Q, epsilon, env.nA)
        cout = 0
        pre_states = []
        converge = False
        # you have to reset the environment first
        state = env.reset()
        frames = []
        while True:
            print(cout)
            pre_states.append(state)
            action_prob = policy(state)
            action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
            next_state, _, done, _ = env.step(action)
            ax, fig = env.showWorld(tlt=title)
            env.movingAgent(ax, state, pre_states, action)
            img = fig_to_image(fig)
            frames.append(img)
            plt.close()
            state = next_state
            if converge == True:
                break
            if done:
                converge = True
            cout += 1
        result = os.path.join(save_folder, "tb_results.gif")
        imageio.mimsave(result, frames, fps=5)


def main():
    # define the envrionment
    windygridworld_env = WindyGridworldEnv()
    # define the agent
    agent = Agent_SARSA(windygridworld_env)
    #
    Q_opt, policy_opt = agent.sarsa(env=windygridworld_env, num_episodes=200)
    #
    ax, _ = windygridworld_env.showWorld(tlt="Windy Grid World Environment")
    policy_show = policy_opt.reshape((7,10,4))
    windygridworld_env.showPolicy(policy_show, ax, savefig=os.path.join(SAVE_PATH, "tb_policy.png"))
    #
    agent.saveExamples(env=windygridworld_env, save_folder=SAVE_PATH)

if __name__ == '__main__':
    main()