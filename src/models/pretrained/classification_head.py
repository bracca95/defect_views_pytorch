from torch import nn

from src.models.model import Model
from src.utils.tools import Logger
from src.utils.config_parser import Config


class Head(Model):

    def __init__(self, config: Config, extractor: Model, out_class: int, freeze: bool = False):
        super().__init__(config)

        self.extractor = extractor
        backbone_features: int = self.extractor.get_out_size(1)
        if freeze:
            Logger.instance().debug("freezing extractor layers")
            list(map(lambda param: setattr(param, 'requires_grad', False), self.extractor.parameters()))

        self.linear = nn.Sequential(
            nn.Linear(backbone_features, out_class),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        out = self.extractor(x)
        return self.linear(out)