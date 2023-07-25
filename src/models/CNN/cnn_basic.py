from torch import nn

from src.models.model import Model
from src.utils.config_parser import Config

class CNN(Model):

    def __init__(self, config: Config):
        super().__init__(config)

        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.seq2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        out = self.seq1(x)
        out = self.seq2(out)

        return out.view(out.size(0), -1)
