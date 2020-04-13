from collections import deque
import os
import math

import torch
import random
import torch.nn as nn
import numpy as np
import gym
import torch.nn.functional as F
import torch.optim as optim

from blokus.envs.blokus_env import BlokusEnv
from segment_tree import SegmentTree


### Models ###
class DQN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, out_dim)
        )

    def forward(self, x):
        return self.layers(x)


class DuelingNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DuelingNetwork, self).__init__()
        self.input_layer = nn.Sequential(nn.Linear(in_dim, 24),
                                         nn.ReLU())
        self.advantage_layer = nn.Sequential(nn.Linear(24, 24),
                                             nn.ReLU(),
                                             nn.Linear(24, out_dim))
        self.value_layer = nn.Sequential(nn.Linear(24, 24),
                                         nn.ReLU(),
                                         nn.Linear(24, 1))

    def forward(self, x):
        x = self.input_layer(x)
        advantage = self.advantage_layer(x)
        value = self.value_layer(x)
        return advantage + value - advantage.mean()


class NoisyNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, sigma_init=0.4):
        super(NoisyNetwork, self).__init__()
        self.input_layer = nn.Sequential(nn.Linear(in_dim, 24),
                                         nn.ReLU())
        self.hidden_noisy_layer = NoisyLayer(24, 24, sigma_init)
        self.output_noisy_layer = NoisyLayer(24, out_dim, sigma_init)

    def update_noise(self):
        self.hidden_noisy_layer.update_noise()
        self.output_noisy_layer.update_noise()

    def forward(self, x):
        x = self.input_layer(x)
        x = nn.ReLU()(self.hidden_noisy_layer(x))
        return self.output_noisy_layer(x)

# Inspired from https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/05.noisy_net.ipynb
class NoisyLayer(nn.Module):
    def __init__(self, in_dim, out_dim, sigma_init):
        super(NoisyLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma_init = sigma_init
        self.mu_w = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.mu_b = nn.Parameter(torch.Tensor(out_dim))
        self.sigma_w = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.sigma_b = nn.Parameter(torch.Tensor(out_dim))
        # Epsilon is not trainable
        self.register_buffer("eps_w", torch.Tensor(out_dim, in_dim))
        self.register_buffer("eps_b", torch.Tensor(out_dim))
        self.init_params()
        self.update_noise()

    def init_params(self):
        # Trainable params
        self.mu_w.data.uniform_(-math.sqrt(1 / self.in_dim), math.sqrt(1 / self.in_dim))
        self.mu_b.data.uniform_(-math.sqrt(1 / self.in_dim), math.sqrt(1 / self.in_dim))
        self.sigma_w.data.fill_(self.sigma_init / math.sqrt(self.out_dim))
        self.sigma_b.data.fill_(self.sigma_init / math.sqrt(self.out_dim))

    def update_noise(self):
        self.eps_w.copy_(self.factorize_noise(self.out_dim).ger(self.factorize_noise(self.in_dim)))
        self.eps_b.copy_(self.factorize_noise(self.out_dim))

    def factorize_noise(self, size):
        # Modify scale to amplify or reduce noise
        x = torch.Tensor(np.random.normal(loc=0.0, scale=0.001, size=size))
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x):
        return F.linear(x, self.mu_w + self.sigma_w * self.eps_w, self.mu_b + self.sigma_b * self.eps_b)


### Memory buffers ###
class ReplayMemory:
    def __init__(self, max_size, batch_size):
        self.max_size = max_size
        self.batch_size = batch_size
        self.memory = deque([], maxlen=max_size)

    def __len__(self):
        return len(self.memory)

    def add_to_memory(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))

    def random_batch(self):
        return random.sample(self.memory, self.batch_size)


# TODO not finishedparame
class PrioritizedExperienceReplay(ReplayMemory):
    def __init__(self, max_size, batch_size, tree_capacity, a):
        super(PrioritizedExperienceReplay, self).__init__(max_size, batch_size)
        self.tree = SegmentTree(tree_capacity)
        self.a = a

    def add_to_memory(self, state, action, next_state, reward, done):
        super().add_to_memory(state, action, next_state, reward, done)

    def get_priority(self, error):
        eps = 0.001
        return (np.abs(error) + eps) ** self.a

    def sample_batch(self):
        segment = self.tree.query(0, len(self) - 1, 'sum') / self.batch_size
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)


### Agent ###
class Agent:
    def __init__(self,
                 env,
                 memory_size,
                 batch_size,
                 learning_rate,
                 num_episodes,
                 model_filename,
                 eps=1,
                 min_eps=0.01,
                 eps_decay=0.005,
                 gamma=0.9,
                 is_double=False,
                 is_dueling=False,
                 is_noisy=False):
        self.env = env
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = ReplayMemory(memory_size, self.batch_size)
        self.eps = eps
        self.min_eps = min_eps
        self.eps_decay = eps_decay
        self.device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
        self.model_path = os.path.join("models", model_filename + ".pt")
        # Blokus
        # self.obs_size = env.observation_space.shape[0] * env.observation_space.shape[1]
        self.obs_size = env.observation_space.n
        self.is_dueling = is_dueling
        self.is_noisy = is_noisy
        if self.is_dueling:
            self.model = DuelingNetwork(self.obs_size, env.action_space.n).to(self.device)
        elif self.is_noisy:
            self.model = NoisyNetwork(self.obs_size, env.action_space.n).to(self.device)
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
            next_action = int(self.model(state).argmax().detach().cpu())

        return next_action

    def update(self, state, target, action):
        prediction = self.model(state)[action]
        loss = F.smooth_l1_loss(prediction, target)
        self.loss.append(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.is_noisy:
            self.model.update_noise()

    def get_target_double(self, next_state):
        # action = self.model(next_state).argmax()
        # return self.model_target(next_state)[action]
        return self.model_target(next_state).max()

    def get_target(self, reward, done, next_state):
        # y = r if done
        target = torch.tensor(reward).to(self.device)
        if not done:
            # y = r + gamma * max Q(s',a') if not done
            if self.is_double:
                next_state_max_Q = self.get_target_double(next_state)
            else:
                next_state_max_Q = self.model(next_state).max()
            target = (next_state_max_Q * self.gamma) + reward
        return target

    def replay(self):
        batch = self.memory.random_batch()
        for state, action, next_state, reward, done in batch:
            target = self.get_target(reward, done, next_state)
            self.update(state, target, action)

    def ohe(self, state):
        ohe_state = torch.zeros(self.env.observation_space.n).to(self.device)
        ohe_state[state] = 1
        return ohe_state

    # def ohe(self, state):
    #     return state.view(-1).type(torch.float32).to(self.device)

    def train(self):
        rewards_lst = []
        best_rate = 0
        for i in range(self.num_episodes):
            rewards = 0
            done = False
            state = self.ohe(self.env.reset())
            while not done:
                action = self.eps_greedy_action(state)
                next_state, reward, done, info = self.env.step(action)
                rewards += reward
                next_state = self.ohe(next_state)
                self.memory.add_to_memory(state, action, next_state, reward, done)

                target = self.get_target(reward, done, next_state)
                self.update(state, target, action)
                self.eps = self.min_eps + (self.eps - self.min_eps) * np.exp(-self.eps_decay * i)

                rewards_lst.append(rewards)
                state = next_state

                if not i % 20 and self.is_double:
                    self.model_target.load_state_dict(self.model.state_dict())

            if len(self.memory) > self.batch_size:
                self.replay()

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
    # env = BlokusEnv()
    memory_size = 1000
    num_episodes = 4000
    batch_size = 32
    # gamma = 0.999
    learning_rate = 0.001
    model_filename = "noisy_frozen_lake"

    agent = Agent(env, memory_size, batch_size, learning_rate, num_episodes, model_filename, is_double=False,
                  is_dueling=False, is_noisy=True)
    agent.train()
    # for i in range(10):
    #     agent.test()

# DQN 4/10 victories
# DQN double 7/10 victories
# Dueling 5/10 victories
