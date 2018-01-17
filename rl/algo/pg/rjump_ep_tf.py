from rl.env.random_jump import RandomJumpEnv
from rl.policy.gaussian_policy_tf import GaussianPolicyTF
from rl.policy.value_estimator_tf import ValueEstimatorTF
from rl.featurizer.rbf_featurizer import RBFFeaturizer
#
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
from collections import namedtuple

EpisodeStats = namedtuple("Stats", [ "episode_rewards"])
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state"])

def actor_critic(env, policy_estimator, value_estimator, num_episodes, discounted_factor=1.0):
    stats = EpisodeStats(episode_rewards=np.zeros(num_episodes))
    for i_episode in range(num_episodes):
        state = env.reset()
        # print(state)
        episode = []
        for t in itertools.count():
            # take a step
            action = policy_estimator.predict(state)
            # print(action)
            next_state, reward, done, _ = env.step(action)
            # next_state = next_state[0][0]
            #
            episode.append(Transition(state=state,
                                      action=action,
                                      reward=reward,
                                      next_state=next_state))
            #
            stats.episode_rewards[i_episode] += reward

            # Calculate TD Target
            # print(state, next_state)
            value_next = value_estimator.predict(next_state)
            td_target = reward + discounted_factor * value_next
            td_error = td_target - value_estimator.predict(state)

            # upate the value estimator
            value_estimator.update(state, td_target)

            # Update the policy estimator
            policy_estimator.update(state, td_error, action)

            # Print out which step we're on, useful for debugging.
            print("\rStep {} @ Episode {}/{} ({})".format(
                t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]), end="")

            #
            if done or t>1000:
                break

            state = next_state
        print("")
    return stats


#
env = RandomJumpEnv()
print("Action space : ", env.action_space)
print("Action space low : ", env.action_space.low[0])
print("Action space high: ", env.action_space.high[0])
print("Observation space : ", env.observation_space)
print("Observation space low : ", env.observation_space.low[0])
print("Observation space high: ", env.observation_space.high[0])

#
num_trails = 5
num_episodes = 100
mean_rewards = np.zeros(shape=(num_trails, num_episodes))
for i in range(num_trails):
    tf.reset_default_graph()
    global_step = tf.Variable(0, name="global_step", trainable=False)
    rbf_featurizer = RBFFeaturizer(env, num_featuries=20)
    policy_estimator = GaussianPolicyTF(env, rbf_featurizer, learning_rate=0.0001)
    value_estimator = ValueEstimatorTF(env, rbf_featurizer, learning_rate=0.01)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        stats = actor_critic(env, policy_estimator, value_estimator, num_episodes, discounted_factor=0.95)
    mean_rewards[i, :] = stats.episode_rewards
    sess.close()

fig = plt.figure()
plt.hold('on')
r_mean = np.mean(mean_rewards,axis=0)
r_std = np.std(mean_rewards, axis=0)
plt.fill_between(range(num_episodes), r_mean - r_std, r_mean + r_std, alpha=0.3)
plt.plot(range(num_episodes), r_mean)
plt.legend(loc='lower right')
plt.show()
