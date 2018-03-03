import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
import os
import imageio
#
from rl.env.cliff_walking import CliffWalkingEnv
from rl.misc.utilies import get_dirs, fig_to_image
from rl.algo.td.sarsa.SARSA import Agent_SARSA
#
ROOT_PATH = os.path.realpath("../../../../")
#
RESULT_PATH = os.path.join(ROOT_PATH, "results")
#
Q_SAVE_PATH = get_dirs(os.path.join(RESULT_PATH, "Qlearning"))
S_SAVE_PATH = get_dirs(os.path.join(RESULT_PATH, "SARSA"))


class Agent_QLearning:

    def __init__(self, env):
        self.Q = defaultdict(lambda : np.zeros(env.nA))
        self.policy = np.ones([env.nS, env.nA]) / env.nA

    def epsilon_greedy_policy(self, env, Q, epsilon=0.1):
        def policy_fn(observation):
            A = np.ones(env.nA, dtype=float) * epsilon / env.nA
            best_action = np.argmax(Q[observation])
            A[best_action] += (1.0 - epsilon)
            return A
        return policy_fn

    def Q_learning(self, env, num_episodes, discount_factor=1.0, alpha=0.5):
        policy = self.epsilon_greedy_policy(env, self.Q)
        for i_episode in range(num_episodes):
            state = env.reset()
            for t in itertools.count():
                action_prob = policy(state)
                action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
                #
                next_state, reward, done, _ = env.step(action)
                #
                best_next_action = np.argmax(self.Q[next_state])
                #
                td_target = reward + discount_factor * self.Q[next_state][best_next_action]
                td_delta = td_target - self.Q[state][action]
                self.Q[state][action] += alpha * td_delta
                #
                if  done:
                    break
                state = next_state
        self.policy = self.optimal_policy(env, self.Q)
        return self.Q, self.policy

    def optimal_policy(self, env, Q):
        for s in range(env.nS):
            best_action = np.argmax(Q[s])
            self.policy[s] = np.eye(env.nA)[best_action]
        return self.policy

    def saveExamples(self, env, title, save_folder=None, epsilon=0.1 ):
        policy = self.epsilon_greedy_policy(env, self.Q, epsilon)
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
            image = fig_to_image(fig)
            frames.append(image)
            plt.close()
            state = next_state
            if converge == True:
                break
            if done:
                converge = True
            cout += 1
        result = os.path.join(save_folder, "tb_cliff.gif")
        imageio.mimsave(result, frames, fps=5)

def main():
    #
    cliffwalking_env = CliffWalkingEnv()
    #
    agent_q = Agent_QLearning(cliffwalking_env)
    agent_s = Agent_SARSA(cliffwalking_env)
    # #
    #
    Q_opt_q, policy_opt_q = agent_q.Q_learning(env=cliffwalking_env, num_episodes=500)
    Q_opt_s, policy_opt_s = agent_s.sarsa(env=cliffwalking_env, num_episodes=500)
    #
    agent_q.saveExamples(env=cliffwalking_env, title="Q Learning", save_folder=Q_SAVE_PATH)
    agent_s.saveExamples(env=cliffwalking_env, title="SARSA", save_folder=S_SAVE_PATH)

if __name__ == '__main__':
    main()



