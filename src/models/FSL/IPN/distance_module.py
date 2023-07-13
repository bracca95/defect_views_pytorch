import torch
import torch.nn.functional as F

from torch import nn


class DistScale(nn.Module):
    
    def __init__(self, in_len: int, out_len: int):
        """Distance Scale module
        
        # HERE Two different types of normalization can be experimented: batch norm and softmax
        """
        
        super().__init__()
        self.linear = nn.Sequential(
            #nn.BatchNorm1d(out_len),
            nn.Softmax(dim=1),
            nn.Flatten(0),
            nn.Linear(in_len, out_len),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.linear(x)
        return out
