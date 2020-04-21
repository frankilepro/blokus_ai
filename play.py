from blokus.envs.blokus_env import BlokusEnv
import gym
import random

if __name__ == "__main__":
    # env = BlokusEnv()
    env = gym.make("blokus:blokus-simple-v0")  # Make sure to do: pip install -e blokus in root
    print(f"number of possible moves {env.action_space}")
    for _ in range(100000):
        done = False
        while True:
            if done:
                # print(env.ai.all_ids_to_move.keys())
                # input()
                print(f"{'won' if reward == 2 else ('tie-won' if reward == 0 else 'lost')}")
                observation = env.reset()
                break

            # env.render("minimal")
            # input()
            action = env.action_space.sample()
            action = random.randint(0, 918)
            observation, reward, done, info = env.step(action)
            # print(reward)

    print(f"Starter won {env.starter_won / env.games_played * 100:.2f}%")
    env.close()
