from __future__ import annotations

import random

import numpy as np
import torch
import gymnasium as gym

from Agent import Agent
from config import *


def main():
    env = gym.make('Humanoid-v4', render_mode="human")

    assert (env.observation_space.shape[0] == OBSERVATIONS_SPACE)
    assert (env.action_space.shape[0] == ACTIONS_SPACE)

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    agent = Agent(OBSERVATIONS_SPACE, ACTIONS_SPACE, lr=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)
    reward_over_episodes = []

    for episode in range(N_ITER):
        obs, info = env.reset(seed=SEED)
        total_reward = 0

        done = False
        while not done:
            action = agent.sample_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            agent.rewards.append(reward)
            total_reward += reward
            done = terminated or truncated

        reward_over_episodes.append(total_reward)
        agent.update()

        if episode % 100 == 0:
            avg_reward = int(np.mean(reward_over_episodes[-50:]))
            print("Episode:", episode, "Average Reward:", avg_reward)


if __name__ == "__main__":
    main()
