""" Trains an agent with (stochastic) Policy Gradients on NChain. Uses OpenAI Gym.
reference: Karpathy's blog
"""

import numpy as np
import gym

#
env = gym.make("NChain-v0")
num_actions = env.action_space.n
num_states = env.observation_space.n

print("number of actions : {}".format(num_actions))
print("number of states : {}".format(num_states))

# hyperparameters
batch_size = 10
learning_rate = 1e-3
gamma = 0.99
decay_rate = 0.99
render = False

# model initialization
inputs = num_states
hidden = 200
outputs = num_actions
#
model = {}
model['W1'] = np.random.randn(inputs, hidden) / np.sqrt(inputs)
model['W2'] = np.random.randn(hidden, outputs) / np.sqrt(hidden)

# update buffers that add up gradients over a batch
grad_buffer = { k : np.zeros_like(v) for k,v in model.items() }
# rmsprop memory
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() }

def featurizer(state):
    state_one_hot = np.zeros(num_states)
    state_one_hot[state] = 1
    return state_one_hot

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def grad_softmax(x):
    return softmax(x) * (1.0 - softmax(x))

def discount_reward(r):
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for t in reversed(range(0,r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(x):
    h = np.dot(x, model['W1'])
    h[h<0] = 0 # backpro ReLu nonlinearity
    logp = np.dot(h, model['W2'])
    p = softmax(logp)
    return p, h

def policy_backward(epx, eph, epdlogp):
    # print(epx.shape, eph.shape, epdlogp.shape)
    dW2 = np.dot(eph.T, epdlogp)
    # print(dW2.shape)
    dh = np.dot(epdlogp, model['W2'].T)
    dh[eph <= 0] = 0
    # print(dh.shape)
    dW1 = np.dot(epx.T, dh)
    # print(dW1.shape)
    return {'W1': dW1, 'W2': dW2}


xs, hs, dlogps, drs, yo = [], [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

observation = env.reset()
while True:
    if render: env.render()
    # sample an action from the returned probability
    x = featurizer(observation)
    aprob, h = policy_forward(x)
    action = np.random.choice(np.arange(len(aprob)), p=aprob)
    # print("state ", observation, "action ", aprob)
    #
    # record various intermediates (needed later for backprop)
    xs.append(x)
    hs.append(h)
    #
    y_one_hot = np.zeros(num_actions)
    y_one_hot[action] = 1
    dlogps.append(y_one_hot-aprob)
    yo.append(aprob)
    # step the environment and get new measurements
    observation, reward, done, _ = env.step(action)
    reward_sum += reward
    drs.append(reward)

    # roll out, an episode finished
    # print(done)
    if done:
        print(np.mean(drs))
        episode_number += 1

        # stack together all inputs, hidden states, action gradients and rewards for this episode
        ep_x = np.vstack(xs)
        ep_h = np.vstack(hs)
        ep_dlogp = np.vstack(dlogps)
        ep_r = np.vstack(drs)
        ep_yo = np.vstack(yo)
        xs, hs, dlogps, drs = [], [], [], []

        # compute the discounted reward backwards through time
        discounted_epr = discount_reward(ep_r)
        # standarize the rewards to be unit normal
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        # # # Modulate the gradient with advantage (PG magic happens right here)
        ep_dlogp *= discounted_epr
        grad = policy_backward(epx=ep_x, eph=ep_h, epdlogp=ep_dlogp)

        #
        for k in model:
            grad_buffer[k] += grad[k]

        if episode_number % batch_size == 0:
            for k, v in model.items():
                g = grad_buffer[k]
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1-decay_rate)* g**2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                # reset batch gradient buffer
                grad_buffer[k] = np.zeros_like(v)
        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        for state in range(num_states):
            features_x = featurizer(state)
            test_aprob, _ = policy_forward(features_x)
            print("state ", state, "action ", test_aprob)
        reward_sum = 0
        observation = env.reset()




