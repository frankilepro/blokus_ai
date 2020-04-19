import os

import gym
import torch.optim as optim

from memory_replay import ReplayMemory, PrioritizedExperienceReplay
from models import *


class Agent:
    """
    :param env: gym environment
    :param memory_size: maximum capacity of the memory replay
    :param batch_size
    :param learning_rate
    :param num_episodes: number of games played to train on
    :param eps: initial epsilon value in the epsilon-greedy policy
    :param min_eps: minimal possible value of epsilon in the epsilon-greedy policy
    :param eps_decay: decrease rate of epsilon in the epsilon-greedy policy
    :param gamma: discount factor used to measure the target
    :param is_double: boolean to have double DQN network
    :param is_dueling: boolean to have dueling network
    :param is_noisy: boolean to have a noisy network
    :param is_distributional: boolean to have a distributional network
    :param dist_params: dictionary containing parameters for distributional network including num_bins (number of bins
                        the distribution return), v_min (minimal state value), v_max (maximal state value)
                        e.i: {num_bin:51, v_min:0, v_max:1} 51 atoms are used in the paper
    """

    def __init__(self,
                 env,
                 memory_size,
                 batch_size,
                 learning_rate,
                 num_episodes,
                 model_filename,
                 nsteps=None,
                 eps=1,
                 min_eps=0.01,
                 eps_decay=0.00005,
                 gamma=0.9,
                 is_double=False,
                 is_dueling=False,
                 is_noisy=False,
                 is_distributional=False,
                 is_prioritized=False,
                 prioritized_params=None,
                 distr_params=None):
        self.env = env
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.batch_size = batch_size
        self.nsteps = nsteps
        if is_prioritized:
            self.prioritized_params = prioritized_params
            self.memory = PrioritizedExperienceReplay(memory_size, self.batch_size, prioritized_params)
        else:
            self.memory = ReplayMemory(memory_size, self.batch_size, self.gamma, self.nsteps)
        self.eps = eps
        self.min_eps = min_eps
        self.eps_decay = eps_decay
        self.device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
        self.model_path = os.path.join("models", model_filename + ".pt")
        # Blokus
        self.obs_size = env.observation_space.shape[0] * env.observation_space.shape[1]
        # self.obs_size = env.observation_space.n
        self.is_dueling = is_dueling
        self.is_noisy = is_noisy
        self.is_distributional = is_distributional
        self.is_prioritized = is_prioritized
        if self.is_distributional:
            self.distr_params = distr_params
            self.distr_params["v_range"] = torch.linspace(self.distr_params["v_min"],
                                                          self.distr_params["v_max"],
                                                          self.distr_params["num_bins"]).to(self.device)
        if self.is_dueling:
            self.model = DuelingNetwork(self.obs_size, env.action_space.n).to(self.device)
        elif self.is_noisy:
            self.model = NoisyNetwork(self.obs_size, env.action_space.n).to(self.device)
        elif self.is_distributional:
            self.model = DistributionalNetwork(self.obs_size, env.action_space.n, self.distr_params).to(self.device)
        else:
            self.model = DQN(self.obs_size, env.action_space.n).to(self.device)
            # self.model = torch.load(self.model_path, map_location=self.device)
        self.is_double = is_double
        self.loss = []
        if self.is_double:
            self.model_target = DQN(self.obs_size, env.action_space.n).to(self.device)
            self.model_target.load_state_dict(self.model.state_dict())
            self.model_target.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def eps_greedy_action(self, state):
        # Explore the environment
        if self.eps > np.random.random():
            # Take a random action
            next_action = self.env.action_space.sample()
        # Greedy choice (exploitation)
        else:
            possible_moves = [self.env.ai_possible_indexes()]
            # possible_moves = [[0, 1, 2, 3]]
            next_action = int(self.model(state.unsqueeze(0), possible_moves).argmax().detach().cpu())

        return next_action

    def update(self, reward, done, next_state, state, action, possible_move, indices=None, weights=None):
        if self.is_distributional:
            loss = self.get_distributional_loss(reward, done, next_state, state, action)
        else:
            target = self.get_target(reward, done, next_state, possible_move)
            prediction = self.model(state, possible_move).gather(1, action)
            reduction = "none" if self.is_prioritized else "mean"
            loss = F.smooth_l1_loss(prediction, target, reduction=reduction)
            if self.is_prioritized:
                loss_no_reduction = loss.clone()
                loss = torch.mean(loss * weights.to(self.device))
                priority = loss_no_reduction.detach().cpu().numpy() + self.prioritized_params["eps"]
                self.memory.update_priorities(indices, priority)

        self.loss.append(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.is_noisy:
            self.model.update_noise()

    # Inspired from https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/06.categorical_dqn.ipynb
    def get_distributional_loss(self, reward, done, next_state, state, action):
        action_distr = self.model.action_distr(state, self.env)
        log_action_distr = action_distr[range(self.batch_size), action.reshape(-1)].log()

        with torch.no_grad():
            d_target = ((1 - done.float()) * self.gamma * self.distr_params["v_range"]
                        + reward).clamp(self.distr_params["v_min"], self.distr_params["v_max"])
            v_step = (self.distr_params["v_max"] - self.distr_params["v_min"]) / (self.distr_params["num_bins"] - 1)
            delta = (d_target - self.distr_params["v_min"]) / v_step

            offset = torch.linspace(0,
                                    (self.batch_size - 1) * self.distr_params["num_bins"],
                                    self.batch_size).to(self.device).repeat(self.distr_params["num_bins"], 1).T

            next_action = self.model(next_state, self.env).argmax(1)
            next_action_distr = self.model.action_distr(next_state, self.env)[range(self.batch_size), next_action]

            # Projection
            distr_projection = torch.zeros(next_action_distr.shape).to(self.device)
            distr_projection.reshape(-1).index_add_(0,
                                                    (delta.floor() + offset).long().reshape(-1),
                                                    (next_action_distr * (delta.ceil() - delta)).reshape(-1))
            distr_projection.reshape(-1).index_add_(0,
                                                    (delta.ceil() + offset).long().reshape(-1),
                                                    (next_action_distr * (delta - delta.floor())).reshape(-1))

        return - (distr_projection * log_action_distr).sum(1).mean()

    def get_target_double(self, next_state, possible_move):
        action = self.model(next_state, possible_move).argmax(1)
        return self.model_target(next_state)[action]

    def get_target(self, reward, done, next_state, possible_move):
        if self.is_double:
            next_state_max_Q = self.get_target_double(next_state, possible_move)
        else:
            next_state_max_Q = self.model(next_state, possible_move).max(dim=1).values.unsqueeze(1)

        # y = r if done, y = r + gamma * max Q(s',a') if not done
        return (1 - done.float()) * next_state_max_Q * self.gamma + reward.float()

    def replay(self):
        if self.is_prioritized:
            state, action, next_state, reward, done, possible_move, idx, weight = self.memory.get_prioritized_sample()
        else:
            idx, weight = None, None
            state, action, next_state, reward, done, possible_move = self.memory.get_random_batch()
        self.update(reward, done, next_state, state, action, possible_move, idx, weight)

    # def ohe(self, state):
    #     ohe_state = torch.zeros(self.obs_size).to(self.device)
    #     ohe_state[state] = 1
    #     return ohe_state

    def ohe(self, state):
        return state.view(-1).type(torch.float32).to(self.device)

    def train(self):
        rewards_lst = []
        best_rate = 0
        for i in range(self.num_episodes):
            rewards = 0
            done = False
            state = self.ohe(self.env.reset())
            while not done:
                action = self.eps_greedy_action(state)
                next_state, reward, done, _ = self.env.step(action)
                possible_move = self.env.ai_possible_indexes()
                # possible_move = [0, 1, 2, 3]
                env.render("human")
                rewards += reward
                next_state = self.ohe(next_state)
                if self.nsteps is not None:
                    self.memory.add_nsteps_memory(state, action, next_state, reward, done, possible_move)
                elif self.is_prioritized:
                    self.prioritized_params["b"] = min(1.0, i / num_episodes) * (1 - self.prioritized_params["b"]) + \
                                                   self.prioritized_params["b"]
                    self.memory.update_beta(self.prioritized_params["b"])

                self.memory.add_to_memory(state, action, next_state, reward, done, possible_move)

                rewards_lst.append(rewards)
                state = next_state

                if not i % 20 and self.is_double:
                    self.model_target.load_state_dict(self.model.state_dict())

                if len(self.memory) > self.batch_size:
                    self.replay()
                    self.eps = self.min_eps + (self.eps - self.min_eps) * np.exp(-self.eps_decay * i)

            if not i % 10 and i != 0:
                print('Episode {} Loss: {} Reward Rate {}'.format(i, self.loss[-1], str(sum(rewards_lst) / i)))
                if (sum(rewards_lst) / i) > best_rate:
                    best_rate = (sum(rewards_lst) / i)
                    torch.save(self.model, self.model_path)
        torch.save(self.model, self.model_path)
        self.env.close()

    def test(self):
        self.model = torch.load(self.model_path, map_location=self.device)
        done = False
        state = self.ohe(self.env.reset())
        rewards = 0
        self.eps = self.min_eps
        while not done:
            action = self.eps_greedy_action(state)
            # self.env.render()
            next_state, reward, done, info = self.env.step(action)
            rewards += reward
            state = self.ohe(next_state)
        if rewards:
            print("Victory")
        else:
            print("Lost")
        self.env.close()


if __name__ == "__main__":
    env = gym.make("FrozenLake-v0")
    # env = gym.make("blokus:blokus-duo-v0")
    memory_size = 1000
    num_episodes = 10000

    batch_size = 32
    # gamma = 0.999
    learning_rate = 0.001
    model_filename = "blokus"

    dist_params = {"num_bins": 51, "v_min": 0.0, "v_max": 1.0}
    prioritized_params = {"a": 0.2, "b": 0.6, "eps": 1e-5}
    agent = Agent(env, memory_size, batch_size, learning_rate, num_episodes, model_filename, nsteps=None,
                  is_double=False, is_dueling=False, is_noisy=False, is_distributional=False, distr_params=dist_params,
                  is_prioritized=False, prioritized_params=prioritized_params)
    agent.train()
    # for i in range(10):
    #     agent.test()
