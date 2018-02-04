from rl.policy import EpsilonGreedyPolicy
from rl.misc.plotting import plot_value_function
from rl.misc.utilies import fig_to_image
from gym.envs.toy_text.nchain import NChainEnv
from collections import defaultdict
import numpy as np
import imageio
import matplotlib.pyplot as plt
#
env = NChainEnv(n=5, slip=0.2)
#
print("Action Space ", env.action_space)
print("Observation Space ", env.observation_space)

def save_value_fn_gif(Q, i_episode):
    V = defaultdict(float)
    for state, actions_value in Q.items():
        action_value = np.max(actions_value)
        V[state] = action_value
    fig = plot_value_function(V, show=False, title="Value funtion. iter: {}".format(i_episode))
    image = fig_to_image(fig)
    plt.close()
    return image


def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, save=False):
    Q = defaultdict(lambda : np.zeros(env.action_space.n))
    policy = EpsilonGreedyPolicy(env, Q, epsilon=0.95)
    # policy = GreedyPolicy(env=env, Q=Q)
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    frames = []
    for i_episode in range(num_episodes):
        rewards = []
        episode = []
        state = env.reset()
        for t in range(200):
            action = policy.predict(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            rewards.append(reward)
            if done:
                break
            state = next_state
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}. Rewards : {}".format(i_episode, num_episodes, np.mean(rewards)), end="\n")
            for key, value in Q.items():
                print("state-action pair", key, value)
            if save:
                frames.append(save_value_fn_gif(Q, i_episode))
        state_action_pairs = set([(x[0], x[1]) for x in episode])
        # print("Episode {}".format(i_episode))
        for state, action in state_action_pairs:
            state_action_pair = (state, action)
            # Find the first occurance of the (state, action) pair in the episode
            first_occurence_idx = next(i for i, x in enumerate(episode)
                                       if x[0] == state and x[1] == action)
            # print("state {}, action {}, idx {}".format(state, action, first_occurence_idx))
            # sum up all rewards since the first occurance
            G = sum([x[2] *(discount_factor**i) for i, x in enumerate(episode[first_occurence_idx:])])
            # Caculate average return for this state over all sampled episodes
            returns_sum[state_action_pair] += G
            returns_count[state_action_pair] += 1.0
            Q[state][action] = returns_sum[state_action_pair] / returns_count[state_action_pair]
        policy.update(Q=Q)
    return Q, policy, frames


if __name__ == '__main__':
    Q, policy, frames = mc_control_epsilon_greedy(env, num_episodes=50000, discount_factor=.9, save=False)
    if len(frames) > 1:
        imageio.mimsave("/Users/yunlong/Gitlab/rl_prototype/results/mc/mc_on_policy_value.gif", frames, fps=5)
    for key, value in Q.items():
        print(key, value)
    # print(V.keys())
