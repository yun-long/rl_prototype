import numpy as np
import imageio
import itertools
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
from collections import namedtuple
#
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
from rl.misc.utilies import get_dirs, fig_to_image
from rl.algo.td.util import plot_cost_mountain_car, plot_episode_stats
from gym.envs.classic_control.mountain_car import MountainCarEnv
#
# Global variables
ROOT_PATH = os.path.realpath("../../../../")
RESULT_PATH = get_dirs(os.path.join(ROOT_PATH, 'results'))
EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

#
class Approximator:
    """
    Value Function Approximator
    """
    def __init__(self, env):
        self.observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        self.scaler = self.standard_scaler(env)
        self.featurizer = self.sklearn_featurizer()
        # I don't quite understand this part.
        self.models = []
        for _ in range(env.action_space.n):
            # Linear model fitted by minimizing a regularized empirical loss with SGD
            model = SGDRegressor(learning_rate="constant")
            # Fit linear model with Stochastic Gradient Descent.
            # X = featurized random state
            # y = [0] ???
            model.partial_fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)

    def featurize_state(self, state):
        """
        represent a state as featurized representation
        :return: 
        """
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]

    def predict(self, s, a=None):
        """
        Makes value function predictions.

        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for

        Returns
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.
        """
        features = self.featurize_state(s)
        if not a:
            return np.array([m.predict([features])[0] for m in self.models])
        else:
            return self.models[a].predict([features])[0]

    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [y])


    def standard_scaler(self, env):
        """
        Used for Normalizing the data to zero mean and unit variance
        :param env: 
        :return: sklearn standard scaler
        """
        # Create scaler for standardizing features by removing the mean and scaling to unit variance
        scaler = sklearn.preprocessing.StandardScaler()
        # Compute the mean and std to be used for later scaling
        scaler.fit(self.observation_examples)
        #
        return scaler

    def sklearn_featurizer(self):
        """
        Used to convert a state to a featurized representation.
        :return: 
        """
        # Concatenates results of multiple transformer objects.
        featurizer = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
        # Transform X separately by each transformer, concatenate results.
        featurizer.fit(self.scaler.transform(self.observation_examples))
        #
        return featurizer

class Agent_QLearning_VFA:
    """
    Agent for Q-Learning algorithm for off-policy TD control with Value Function Approximation.
    Finds the optimal greedy policy while following an epsilon greedy policy 
    """
    def __init__(self, approximator):
        # Because we are using function approximation, there are no such arrays for storing
        # state values or state-action values as tabular methods.
        self.V = None
        self.Q = None
        # instead we use function Approximator to approximate value function
        self.approximator = approximator

    def epsilon_greedy_policy(self, nA, epsilon=0.1):
        """
        Epsilon Greedy policy, for trade-off between exploitation and and exploration 
        :param epsilon: 
        :param nA: number of actions
        :return: policy function
        """
        def policy_fn(observation):
            A = np.ones(nA, dtype=float) * epsilon/ nA
            # Different to tabular method, we have approximator which stores paramters,
            # instead tables that stores state values or action-state values.
            q_values = self.approximator.predict(observation)
            best_action = np.argmax(q_values)
            A[best_action] += 1 - epsilon
            return A
        return policy_fn

    def q_learning_fa(self,env, num_episodes, discount_factor=1.0, epsilon=0.1):
        frames = []
        episode_stats = EpisodeStats(episode_lengths=np.zeros(num_episodes),
                                     episode_rewards=np.zeros(num_episodes))
        for i_episode in range(num_episodes):
            print("Episode {0}".format(i_episode))
            # The policy we are following
            policy = self.epsilon_greedy_policy(nA=env.action_space.n, epsilon=epsilon)
            # Reset the environment and pick the first action
            state = env.reset()
            if i_episode % 5 == 0:
                fig = plot_cost_mountain_car(env, self.approximator, step=i_episode)
                image = fig_to_image(fig)
                frames.append(image)
                plt.close()
            for t in itertools.count():
                # Choose an action
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                # Take a step.
                next_state, reward, done, _ = env.step(action)
                # store staatistics
                episode_stats.episode_lengths[i_episode] = t
                episode_stats.episode_rewards[i_episode] += reward
                # TD Update.
                q_values_next = self.approximator.predict(next_state)
                # compute TD target.
                td_target = reward + discount_factor * np.max(q_values_next)
                # update the parameters in function approximator.
                self.approximator.update(state, action, td_target)
                if done:
                    break
                state = next_state
        save_path = get_dirs(os.path.join(RESULT_PATH, "Qlearning"))
        imageio.mimsave(save_path + "/vfa_values_qlearning.gif", frames, fps=20)
        return episode_stats

    def run_policy(self, env):
        frames = []
        state = env.reset()
        policy = self.epsilon_greedy_policy(env.action_space.n)
        for _ in itertools.count():
            frames.append(env.render(mode='rgb_array'))
            action_prob = policy(state)
            action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
            next_step, _, done, _ = env.step(action)
            if done:
                env.close()
                break
            state = next_step
        save_path = get_dirs(os.path.join(RESULT_PATH, "Qlearning"))
        imageio.mimsave(save_path + "/vfa_car_qlearning.gif", frames, fps=30)

def main():
    # create a Mountain Car environment
    env = MountainCarEnv()
    # create a approximator for the agent
    approximator = Approximator(env)
    # create an agent that knows how to perform Q learing method with value function approximation method
    agent_qlearning_vfa = Agent_QLearning_VFA(approximator)
    # perform q learning with function approximation method
    Q_stats = agent_qlearning_vfa.q_learning_fa(env=env, num_episodes=200)
    #
    plot_episode_stats(stats=Q_stats)
    save_path = get_dirs(os.path.join(RESULT_PATH, "Qlearning"))
    plt.savefig(save_path+ "/vfa_rewards.png")
    # plt.show()
    #
    agent_qlearning_vfa.run_policy(env)



if __name__ == '__main__':
    main()