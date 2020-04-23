import blokus
from blokus.envs.blokus_env import BlokusEnv
import gym
import random
import pkg_resources

if __name__ == "__main__":
    # version = pkg_resources.get_distribution("compare_with_remote").version
    # split_version = version.split('.')
    # try:
    #     split_version[-1] = str(int(split_version[-1]) + 1)
    # except ValueError:
    #     # do something about the letters in the last field of version
    #     pass
    # new_version = '.'.join(split_version)
    new_version = float(pkg_resources.get_distribution("blokus").version) + 0.01
    print(f"{new_version:0.2f}")
    # # env = BlokusEnv()
    # env = gym.make("blokus:blokus-hard-greedy-v0")  # Make sure to do: pip install -e blokus in root
    # print(f"number of possible moves {env.action_space}")
    # for _ in range(100000):
    #     while True:
    #         # input()
    #         action = env.action_space.sample()
    #         # action = random.randint(0, 918)
    #         observation, reward, done, info = env.step(action)
    #         env.render("human")
    #         # print(env.ai.all_ids_to_move.keys())
    #         print(reward)

    #         if done:
    #             # print(env.ai.all_ids_to_move.keys())
    #             # input()
    #             print(f"{'won' if reward == 2 else ('tie-won' if reward == 0 else 'lost')}")
    #             # print(env.ai.all_ids_to_move.keys())
    #             observation = env.reset()
    #             # input()
    #             break

    # print(f"Starter won {env.starter_won / env.games_played * 100:.2f}%")
    # env.close()
