import random
from collections import deque

import torch
import numpy as np
from segment_tree import SegmentTree


class ReplayMemory:
    def __init__(self, max_size, batch_size, gamma, nsteps=None):
        self.max_size = max_size
        self.batch_size = batch_size
        self.memory = deque([], maxlen=max_size)
        self.nsteps = nsteps
        self.nsteps_buffer = deque([], maxlen=nsteps)
        self.gamma = gamma

    def __len__(self):
        return len(self.memory)

    def add_to_memory(self, state, action, next_state, reward, done, possible_move):
        self.memory.append((state, action, next_state, reward, done, possible_move))

    def add_nsteps_memory(self, state, action, next_state, reward, done, possible_move):
        self.nsteps_buffer.append((state, action, next_state, reward, done, possible_move))
        if len(self.nsteps_buffer) < self.nsteps:
            return

        next_state, nsteps_reward, done = self.nsteps_buffer[-1][2:5]
        for i in range(self.nsteps - 2, -1, -1):
            ns, r, d = self.nsteps_buffer[i][2:5]
            nsteps_reward = nsteps_reward * (self.gamma ** i) * (1 - d) + r
            if d:
                next_state = ns
                done = d
        state, action = self.nsteps_buffer[0][:2]
        possible_move = self.nsteps_buffer[0][-1]
        self.add_to_memory(state, action, next_state, nsteps_reward, done, possible_move)

    def random_batch(self):
        random_batch = random.sample(self.memory, self.batch_size)
        return self.create_batch(random_batch)

    def create_batch(self, random_batch):
        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []
        possible_moves = []
        for b in random_batch:
            state, action, next_state, reward, done, move = b
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            possible_moves.append(move)
        states = torch.cat(states).reshape(self.batch_size, -1)
        actions = torch.tensor(actions).unsqueeze(1).to(states.device)
        next_states = torch.cat(next_states).reshape(self.batch_size, -1)
        rewards = torch.tensor(rewards).unsqueeze(1).to(states.device)
        dones = torch.tensor(dones).unsqueeze(1).to(states.device)
        return states, actions, next_states, rewards, dones, possible_moves


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
