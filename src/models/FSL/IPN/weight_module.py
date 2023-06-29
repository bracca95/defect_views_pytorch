import torch
from torch import nn


class Weight(nn.Module):

    def __init__(self, in_len: int, k_shot: int):
        """
        in_len should be hidden dimension output from model * number of k_shot
        """
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_len, k_shot),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # build prototype
        weight = self.linear(x)
        full_weight = weight.unsqueeze(2).expand(-1, weight.shape[-1], x.shape[-1] // weight.shape[-1])

        x = x.reshape_as(full_weight)
        wx = full_weight * x
        return torch.sum(wx, dim=1)