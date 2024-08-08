import torch.nn as nn
import torch


class NetWork(nn.Module):

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        super().__init__()

        hidden_space1 = 512
        hidden_space2 = 1024
        hidden_space3 = 2048

        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
            nn.Linear(hidden_space2, hidden_space3),
            nn.Tanh(),
        )

        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space3, action_space_dims)
        )

        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space3, action_space_dims)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )

        return action_means, action_stddevs
