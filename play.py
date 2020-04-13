import gym
import random


env = gym.make("blokus:blokus-v0")  # Make sure to do: pip install -e blokus in root
observation = env.reset()
while True:
    env.render("minimal")
    actions = env.ai_possible_ids()
    print(len(actions))
    if len(actions) == 0:
        actions = [None]
        print("WTF")

    observation, reward, done, info = env.step(random.choice(actions))

    if done:
        print("game done")
        observation = env.reset()
env.close()
