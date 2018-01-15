from rl.env.random_jump import RandomJumpEnv

env = RandomJumpEnv()

obs = env.reset()
while True:
    action = env.action_space.sample()
    # print(action)
    next_obs, reward, done, _ = env.step(action)
    print(reward)
    env.render()