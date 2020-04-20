from blokus.envs.blokus_env import BlokusEnv
import gym

if __name__ == "__main__":
    # env = BlokusEnv()
    env = gym.make("blokus:blokus-simple-v0")  # Make sure to do: pip install -e blokus in root
    observation = env.reset()
    for _ in range(1000):
        while True:
            action = env.action_space.sample()

            observation, reward, done, info = env.step(action)
            env.render("human")
            # print(reward)

            if done:
                print(env.ai.all_ids_to_move.keys())
                input()
                # print(f"{'won' if reward == 2 else ('tie-won' if reward == 0 else 'lost')}")
                observation = env.reset()
                break

    print(f"Starter won {env.starter_won / env.games_played * 100:.2f}%")
    env.close()
