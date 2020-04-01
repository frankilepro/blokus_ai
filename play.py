import gym
env = gym.make("blokus:blokus-v0")  # Make sure to do: pip install -e blokus in root
observation = env.reset()
for _ in range(100):
    env.render()
    action = env.action_space.sample()  # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()
env.close()
