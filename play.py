import random
import gym
from blokus.envs.blokus_env import BlokusEnv

if __name__ == "__main__":
    # env = BlokusEnv()
    env = gym.make("blokus:blokus-simple-v0")  # Make sure to do: pip install -e blokus in root
    observation = env.reset()
    for _ in range(1000):
        while True:
            # env.render("human")
            action = env.action_space.sample()

            observation, reward, done, info = env.step(action)
            # print(reward)

            if done:
                # print(f"{'won' if reward == 2 else ('tie-won' if reward == 0 else 'lost')}")
                observation = env.reset()
                break
    env.close()
