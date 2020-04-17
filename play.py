import gym
import random


env = gym.make("blokus:blokus-v0")  # Make sure to do: pip install -e blokus in root
observation = env.reset()
while True:
    env.render("human")
    action = env.action_space.sample()

    observation, reward, done, info = env.step(action)

    if done:
        print(f"game done with reward {reward}")
        observation = env.reset()
env.close()
