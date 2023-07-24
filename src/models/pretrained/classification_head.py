import torch
from torch import nn

from src.models.model import Model
from src.models.pretrained.extractors import TimmFeatureExtractor


class Head(Model):

    def __init__(self, extractor: Model, out_class: int):
        super().__init__()

        # TODO: adapt to any type of extractor. ATM it works only with TimmFeatureExtractor
        t = torch.randn(2, 1, 224, 224)
        in_features = extractor(t).size(1) # (batch_size, feat, Opt[unpooled_size], Opt[unpooled_size])

        self.extractor = extractor
        self.linear = nn.Sequential(
            nn.Linear(in_features, out_class),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        out = self.extractor(x)
        return self.linear(out)