import random
from collections import deque

import torch
import numpy as np
from segment_tree import SegmentTree, MinSegmentTree, SumSegmentTree


class ReplayMemory:
    def __init__(self, max_size, batch_size, gamma=0.9, nsteps=None):
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

    def get_random_batch(self):
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


# Inspired from https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/03.per.ipynb
class PrioritizedExperienceReplay(ReplayMemory):
    def __init__(self, max_size, batch_size, prioritized_params):
        super(PrioritizedExperienceReplay, self).__init__(max_size, batch_size)
        self.max_size = max_size
        self.tree_capacity = 1
        while self.tree_capacity < self.max_size:
            # Data structure requires capacity to be a power of 2
            self.tree_capacity *= 2
        self.sum_tree = SumSegmentTree(self.tree_capacity)
        self.min_tree = MinSegmentTree(self.tree_capacity)
        self.a = prioritized_params["a"]
        self.b = prioritized_params["b"]
        self.tree_idx = 0
        self.max_priority = 1.0

    def add_to_memory(self, state, action, next_state, reward, done, possible_move):
        super().add_to_memory(state, action, next_state, reward, done, possible_move)
        self.sum_tree[self.tree_idx] = self.max_priority ** self.a
        self.min_tree[self.tree_idx] = self.max_priority ** self.a
        self.tree_idx = (self.tree_idx + 1) % self.max_size

    def update_priorities(self, indices, priorities):
        eps = 1e-5
        for i, priority in zip(indices, priorities):
            self.sum_tree[i] = (priority + eps) ** self.a
            self.min_tree[i] = (priority + eps) ** self.a
            self.max_priority = max(self.max_priority, priority + eps)

    def sample_uniform(self):
        segment = self.sum_tree.sum(0, len(self) - 1) / self.batch_size
        indices = []
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            mass = random.uniform(a, b)
            idx = self.sum_tree.find_prefixsum_idx(mass)
            indices.append(idx)
        return indices

    def update_beta(self, beta):
        self.b = beta

    def get_prioritized_sample(self):
        indices = self.sample_uniform()
        weights = []
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-self.b)
        batch = []
        for i in indices:
            p = self.sum_tree[i] / self.sum_tree.sum()
            weights.append((p * len(self)) ** (-self.b) / max_weight)
            batch.append(self.memory[i])
        states, actions, next_states, rewards, dones, possible_moves = self.create_batch(batch)
        return states, actions, next_states, rewards, dones, possible_moves, indices, torch.tensor(weights)
