import gym
from rainbow.agent import Agent
import pandas as pd

if __name__ == "__main__":
    env = gym.make("blokus_gym:blokus-simple-v0")
    memory_size = 1000
    num_episodes = 11
    batch_size = 32
    learning_rate = 0.001
    model_filename = "blokus-train5"

    dist_params = {"num_bins": 51, "v_min": -1.0, "v_max": 1.0}
    prioritized_params = {"a": 0.6, "b": 0.6, "eps": 1e-5}

    config_rainbow = {
        "is_double": True,
        "is_dueling": True,
        "is_prioritized": True,
        "is_noisy": True,
        "is_distributional": True,
        "nsteps": 3
    }
    config_dqn = {
        "is_double": False,
        "is_dueling": False,
        "is_prioritized": False,
        "is_noisy": False,
        "is_distributional": False,
        "nsteps": None
    }

    names = ["No Double", "No Dueling", "No PER", "No Noisy", "No Distributional", "No N-steps"]
    results = {}

    for idx, key in enumerate(config_rainbow.keys()):
        config = config_rainbow.copy()
        if key == "nsteps":
            config[key] = None

        print(config)
        agent = Agent(env, memory_size, batch_size, learning_rate, num_episodes, model_filename,
                      nsteps=config["nsteps"], is_double=config["is_double"], is_dueling=config["is_dueling"],
                      is_noisy=config["is_noisy"], is_distributional=config["is_distributional"],
                      distr_params=dist_params, is_prioritized=config["is_prioritized"],
                      prioritized_params=prioritized_params)

        scores = agent.train()
        results[names[idx]] = scores
        scores_df = pd.DataFrame(results, columns=names)
        scores_df.to_csv("blokus-training2.csv")
        scores_df = scores_df.reset_index()

    names = ["Rainbow", "DQN"]
    for idx, config in enumerate([config_rainbow, config_dqn]):
        print(config)
        agent = Agent(env, memory_size, batch_size, learning_rate, num_episodes, model_filename,
                      nsteps=config["nsteps"], is_double=config["is_double"], is_dueling=config["is_dueling"],
                      is_noisy=config["is_noisy"], is_distributional=config["is_distributional"],
                      distr_params=dist_params, is_prioritized=config["is_prioritized"],
                      prioritized_params=prioritized_params)
        scores = agent.train()
        results[names[idx]] = scores
        scores_df = pd.DataFrame(results, columns=names)
        scores_df.to_csv("blokus-training2.csv")
