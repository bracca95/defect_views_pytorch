import timm
import torch

from typing import Optional, List
from src.models.model import Model
from src.utils.tools import Logger


class TimmFeatureExtractor(Model):
    """Manage feature extraction with timm pretrained models

    Models are stored in ~/.cache/torch/hub/checkpoints/
    Remember to clean this folder from time to time.

    See Also:
        https://huggingface.co/docs/timm/models
        https://huggingface.co/docs/timm/feature_extraction
        https://timm.fast.ai/models#My-dataset-doesn't-consist-of-3-channel-images---what-now?
        https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055#0583
        https://towardsdatascience.com/5-most-well-known-cnn-architectures-visualized-af76f1f0065e#6702
        https://stackoverflow.com/a/62118437
    """

    def __init__(self, name: str, in_chans: int, pooled: bool, mean: Optional[List[float]]=None, std: Optional[List[float]]=None):
        super().__init__()
        self.name = name
        self.in_chans = in_chans
        self.pooled = pooled
        self.mean = mean
        self.std = std

        # most likely to be true in our case
        if self.pooled:
            self.m = timm.create_model(self.name, pretrained=True, in_chans=self.in_chans, num_classes=0)
        else:
            self.m = timm.create_model(self.name, pretrained=True, in_chans=self.in_chans, num_classes=0, global_pool='')

        Logger.instance().debug(f"model info: {self.m.default_cfg}")

        if self.mean is not None and self.std is not None:
            Logger.instance().debug(f"using custom mean {self.mean} and std {self.std}")
            self.m.mean = tuple(self.mean)
            self.m.std = tuple(self.std)

    def forward(self, x):
        return self.m(x)