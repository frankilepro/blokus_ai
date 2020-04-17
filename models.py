import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DQN, self).__init__()

        self.layers = nn.Sequential(nn.Linear(in_dim, 24),
                                    nn.ReLU(),
                                    nn.Linear(24, 24),
                                    nn.ReLU(),
                                    nn.Linear(24, out_dim))
        # Softamx only on valid moves
        self.custom_softmax = LegalSoftmax()

    def forward(self, x, env):
        x = self.layers(x)
        return self.custom_softmax(x, env)


class LegalSoftmax(nn.Module):
    """
    Custom layer to consider only valid moves
    """

    def __init__(self):
        super(LegalSoftmax, self).__init__()

    def forward(self, x, env):
        # legal_moves = env.ai_possible_indexes()
        legal_moves = [0, 1, 2, 3]
        actions_tensor = torch.zeros(x.shape).to(x.device)
        actions_tensor[legal_moves] = 1
        filtered_actions = x * actions_tensor
        return F.softmax(filtered_actions, dim=0)


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
        nn.init.uniform_(self.mu_w, -math.sqrt(1 / self.in_dim), math.sqrt(1 / self.in_dim))
        nn.init.uniform_(self.mu_b, -math.sqrt(1 / self.in_dim), math.sqrt(1 / self.in_dim))
        nn.init.constant_(self.sigma_w, self.sigma_init / math.sqrt(self.out_dim))
        nn.init.constant_(self.sigma_b, self.sigma_init / math.sqrt(self.out_dim))

    def update_noise(self):
        self.eps_w.copy_(self.factorize_noise(self.out_dim).ger(self.factorize_noise(self.in_dim)))
        self.eps_b.copy_(self.factorize_noise(self.out_dim))

    def factorize_noise(self, size):
        # Modify scale to amplify or reduce noise
        x = torch.Tensor(np.random.normal(loc=0.0, scale=0.001, size=size))
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x):
        return F.linear(x, self.mu_w + self.sigma_w * self.eps_w, self.mu_b + self.sigma_b * self.eps_b)


class DistributionalNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, distr_params):
        super(DistributionalNetwork, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_bins = distr_params.num_bins
        self.v_range = distr_params.v_range
        self.layers = nn.Sequential(nn.Linear(in_dim, 24),
                                    nn.ReLU(),
                                    nn.Linear(24, 24),
                                    nn.ReLU(),
                                    nn.Linear(24, self.out_dim * self.num_bins))

    def forward(self, x):
        x = self.layers(x)
        x = x.reshape(-1, self.out_dim, self.num_bins)
        x = nn.Softmax(dim=2)(x).clamp(1e-5)
        return (x * self.v_range).sum(dim=2)
