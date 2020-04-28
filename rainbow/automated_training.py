import gym
import pandas as pd

from rainbow.agent import Agent

if __name__ == "__main__":
    env = gym.make("blokus_gym:blokus-simple-v0")
    memory_size = 1000
    num_episodes = 5000
    batch_size = 32
    learning_rate = 0.001
    model_filename = "blokus-train"

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

    configs = [config_rainbow, config_dqn]
    for idx, key in enumerate(config_rainbow.keys()):
        config = config_dqn.copy()
        config[key] = True
        if key == "nsteps":
            config[key] = 3
        configs.append(config)

    for idx, key in enumerate(config_rainbow.keys()):
        config = config_rainbow.copy()
        config[key] = False
        if key == "nsteps":
            config[key] = None
        configs.append(config)

    conf_names = ["Rainbow", "DQN", "Double", "Dueling", "PER", "Noisy", "Distributional", "N-Steps", "No Double",
                  "No Dueling", "No PER", "No Noisy", "No Distributional", "No N-steps"]

    results = {}

    for idx, config in enumerate(configs):
        print(config)
        agent = Agent(env, memory_size, batch_size, learning_rate, num_episodes, model_filename,
                      nsteps=config["nsteps"], is_double=config["is_double"], is_dueling=config["is_dueling"],
                      is_noisy=config["is_noisy"], is_distributional=config["is_distributional"],
                      distr_params=dist_params, is_prioritized=config["is_prioritized"],
                      prioritized_params=prioritized_params)
        scores = agent.train()
        results[conf_names[idx]] = scores
        scores_df = pd.DataFrame(results, columns=conf_names)
        scores_df.to_csv("results_all.csv")
