import gym
import random


env = gym.make("blokus:blokus-v0")  # Make sure to do: pip install -e blokus in root
observation = env.reset()
while True:
    env.render("minimal")
    action = env.action_space.sample()

    observation, reward, done, info = env.step(action)
    print(reward)

    if done:
        print(f"{'won' if reward == 1 else 'lost'}")
        observation = env.reset()
env.close()
