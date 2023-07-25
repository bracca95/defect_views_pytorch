import torch.nn as nn

from src.models.model import Model

def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

def attention(in_dim: int) -> nn.Module:
    return nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )


class ProtoNet(Model):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    '''
    def __init__(self, config, x_dim=1, hid_dim=64, z_dim=64):
        super().__init__(config)
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.att = attention(z_dim)

    def forward(self, x):
        x = self.encoder(x)
        a = self.att(x)
        x = a * x
        return x.view(x.size(0), -1) # with img_size 105, output size is: (batch_size, 64*6*6)