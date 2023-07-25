from torch import nn
from src.models.model import Model


class MLP(Model):
    def __init__(self, config, in_dim: int, out_dim: int = 6):
        super().__init__(config)
        
        ratio = in_dim // 4

        self.hidden1 = nn.Sequential(
            nn.Linear(in_dim, 3*ratio),
            nn.BatchNorm1d(3*ratio),
            nn.ReLU(),
        )
        
        self.hidden2 = nn.Sequential(
            nn.Linear(3*ratio, 2*ratio),
            nn.BatchNorm1d(2*ratio),
            nn.ReLU(),
        )

        self.hidden3 = nn.Sequential(
            nn.Linear(2*ratio, ratio),
            nn.BatchNorm1d(ratio),
            nn.ReLU(),
        )

        self.out = nn.Sequential(
            nn.Linear(ratio, out_dim),
            nn.Softmax(),
        )
            

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        return x