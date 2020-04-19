import random
import gym
from blokus.envs.blokus_env import BlokusEnv

# env = BlokusEnv()
env = gym.make("blokus:blokus-duo-v0")  # Make sure to do: pip install -e blokus in root
observation = env.reset()
for _ in range(10):
    while True:
        # env.render("human")
        action = env.action_space.sample()

        observation, reward, done, info = env.step(action)
        # print(reward)

        if done:
            print(f"{'won' if reward == 1 else 'lost'}")
            observation = env.reset()
            break
env.close()
