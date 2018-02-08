import numpy as np
from collections import defaultdict
from gym.envs.toy_text.nchain import NChainEnv
from rl.policy.discrete_policy import RandomPolicy, GreedyPolicy
#
# env = BlackjackEnv()
env = NChainEnv()
#
print("Action Space : ", env.action_space)
print("Observation Space : ", env.observation_space)
#
def mc_control_importance_sampling(env, num_episodes, behavior_policy, discount_factor=1.0):
    Q = defaultdict(lambda : np.zeros(env.action_space.n))
    C = defaultdict(lambda : np.zeros(env.action_space.n))
    target_policy = GreedyPolicy(env, Q)

    for i_episodes in range(1, num_episodes+1):
        if i_episodes % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episodes, num_episodes), end="")
            for key, value in Q.items():
                print(" ", key, "    ", value)
            # sys.stdout.flush()
        episode = []
        state = env.reset()
        for t in range(100):
            action = behavior_policy.predict(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        # Sum of discounted returns
        G = 0.0
        #
        W = 1.0
        # For each step in episode, backwards
        for t in reversed(np.arange(len(episode))):
            state, action, reward = episode[t]
            G = discount_factor * G + reward
            C[state][action] += W
            #
            alpha = W / C[state][action]
            Q[state][action] += alpha * (G - Q[state][action])
            target_policy.update(Q)
            if action != target_policy.predict(state):
                break
            W = W * 1. / behavior_policy.action_probs[action]

    return Q, target_policy

#
random_policy = RandomPolicy(env)
Q, policy = mc_control_importance_sampling(env, num_episodes=500000, behavior_policy=random_policy,discount_factor=1.0)

