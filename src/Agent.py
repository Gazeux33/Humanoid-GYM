import torch
from torch.distributions import Normal

from Network import NetWork
import numpy as np


class Agent:

    def __init__(self, obs_space_dims: int, action_space_dims: int, lr: float, gamma: float, epsilon: float):
        self.learning_rate = lr
        self.gamma = gamma
        self.eps = epsilon

        self.probs = []
        self.rewards = []

        self.nn = NetWork(obs_space_dims, action_space_dims)
        self.optimizer = torch.optim.AdamW(self.nn.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> float:
        state = torch.tensor(np.array([state]))
        action_means, action_stddevs = self.nn(state)
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)

        action = action.numpy()

        self.probs.append(prob)

        return action

    def update(self):
        running_g = 0
        gs = []

        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        loss = 0
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.probs = []
        self.rewards = []
