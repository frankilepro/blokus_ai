import gym
from rainbow.agent import Agent

if __name__ == "__main__":
    env = gym.make("blokus_gym:blokus-simple-v0")
    memory_size = 1000
    num_episodes = 20
    batch_size = 32
    learning_rate = 0.001
    model_filename = "blokus-greedy"

    dist_params = {"num_bins": 51, "v_min": -1.0, "v_max": 1.0}
    prioritized_params = {"a": 0.6, "b": 0.6, "eps": 1e-5}
    agent = Agent(env, memory_size, batch_size, learning_rate, num_episodes, model_filename, nsteps=3,
                  is_double=True, is_dueling=True, is_noisy=True, is_distributional=True, distr_params=dist_params,
                  is_prioritized=True, prioritized_params=prioritized_params)
    agent.test()
