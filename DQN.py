import torch
import random
import torch.nn as nn
import numpy as np
import gym
import torch.nn.functional as F
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        return self.layers(x)


class ReplayMemory:
    def __init__(self, max_size, batch_size):
        self.max_size = max_size
        self.batch_size = batch_size
        self.memory = []

    def __len__(self):
        return len(self.memory)

    def add_to_memory(self, state, action, next_state, reward, done):
        if len(self.memory) < self.max_size:
            self.memory.append((state, action, next_state, reward, done))

    def random_batch(self):
        return random.sample(self.memory, self.batch_size)


class Agent:
    def __init__(self,
                 env,
                 memory_size,
                 batch_size,
                 learning_rate,
                 num_episodes,
                 eps=1,
                 gamma=0.99):
        self.env = env
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = ReplayMemory(memory_size, self.batch_size)
        self.eps = eps
        self.device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
        self.model = DQN(env.observation_space.n, env.action_space.n).to(self.device)
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

    def update(self, state, target):
        prediction = self.model(state)[0]
        loss = F.smooth_l1_loss(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def replay(self):
        batch = self.memory.random_batch()
        for state, action, next_state, reward, done in batch:
            target = reward
            if not done:
                next_state_max_Q = self.model(next_state).max()
                target = (next_state_max_Q * self.gamma) + reward
            self.update(state, target)

    def ohe(self, state):
        ohe_state = torch.zeros(self.env.observation_space.n).to(self.device)
        ohe_state[state] = 1
        return ohe_state

    def train(self):
        for i in range(self.num_episodes):
            state = self.ohe(env.reset())
            rewards = 0
            done = False
            while not done:
                action = self.eps_greedy_action(state)
                next_state, reward, done, info = self.env.step(action)
                next_state = self.ohe(next_state)
                next_state_max_Q = self.model(next_state).max()
                if len(self.memory) > self.batch_size:
                    self.replay()
                target = (next_state_max_Q * self.gamma) + reward
                self.update(state, target)
                self.eps = self.eps * 0.99
                rewards += reward
                self.memory.add_to_memory(state, action, next_state, reward, done)
                state = next_state
        self.env.close()


if __name__ == "__main__":
    env = gym.make("FrozenLake-v0")
    memory_size = 64
    num_episodes = 5
    batch_size = 32
    gamma = 0.99
    learning_rate = 0.001

    agent = Agent(env, memory_size, batch_size, learning_rate, num_episodes)
    agent.train()