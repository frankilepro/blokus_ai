import blokus_gym
from blokus_gym.envs.blokus_env import BlokusEnv
import gymnasium as gym
import random

if __name__ == "__main__":
    # env = BlokusEnv()
    env = gym.make("blokus_gym:blokus-custom-v0")  # Make sure to do: pip install -e . in root
    print(f"number of possible moves {env.action_space}")
    count = 0
    nb_rounds = 0
    for _ in range(100):
        while True:
            # input()
            action = env.action_space.sample()
            count += len(env.ai_possible_indexes())
            nb_rounds += 1
            # action = random.randint(0, 918)
            observation, reward, done, info = env.step(action)
            env.render("human")
            # print(env.ai.all_ids_to_move.keys())
            # print(reward)

            if done:
                # print(env.ai.all_ids_to_move.keys())
                # input()
                print(f"{'won' if reward == 1 else ('tie-won' if reward == 0 else 'lost')}")
                # print(env.ai.all_ids_to_move.keys())
                observation = env.reset()
                # input()
                break

    print(f"Average number of moves per turn: {count / nb_rounds:.2f}")
    print(f"Starter won {env.starter_won / env.games_played * 100:.2f}%")
    env.close()
