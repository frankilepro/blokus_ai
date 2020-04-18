import random
import gym
from blokus.envs.blokus_env import BlokusEnv
# import pyximport
# pyximport.install(pyimport=True)

env = BlokusEnv()
# env = gym.make("blokus:blokus-v0")  # Make sure to do: pip install -e blokus in root
observation = env.reset()
for _ in range(1000):
    # while True:
    # env.render("human")
    action = env.action_space.sample()

    observation, reward, done, info = env.step(action)
    # print(reward)

    if done:
        print(f"{'won' if reward == 1 else 'lost'}")
        observation = env.reset()
env.close()
