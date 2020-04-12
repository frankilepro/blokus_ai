import gym
import random


env = gym.make("blokus:blokus-v0")  # Make sure to do: pip install -e blokus in root
observation = env.reset()
# for _ in range(100):
done = False
while not done:
    env.render()
    action = random.choice(env.ai_possible_moves())  # your agent here (this takes random actions)
    # action = env.action_space.sample()  # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    # observation, reward, done, info = env.step(None)

    if done:
        observation = env.reset()
env.close()
