from torch import nn

from src.models.MLP.mlp_basic import MLP
from src.models.CNN.cnn_basic import CNN, CNNCompare
from src.models.CNN.cnn_105 import CNN105, CNN105Compare
from src.models.FSL.ProtoNet.protonet import ProtoNet
from src.models.pretrained.extractors import TimmFeatureExtractor
from src.utils.config_parser import Config
    

class ModelBuilder:

    @staticmethod
    def load_model(config: Config, out_classes) -> nn.Module:
        if config.fsl.model.lower() == "default":
            return ProtoNet()
        elif config.fsl.model.lower() == "mlp":
            return MLP(config.image_size * config.image_size)
        elif config.fsl.model.lower() == "cnn":
            return CNN()
        elif config.fsl.model.lower() == "cnn105":
            return CNN105()
        elif config.fsl.model.lower() in ("resnet50", "hrnet_w18", "vit_tiny_patch16_224"):
            return TimmFeatureExtractor(config.fsl.model.lower(), in_chans=1, pooled=True, mean=config.dataset_mean, std=config.dataset_std)
        elif config.fsl.model.lower() == "cnncompare":
            return CNNCompare(out_classes)
        elif config.fsl.model.lower() == "cnn105compare":
            return CNN105Compare(out_classes)
        else:
            raise ValueError(f"fsl.model must be { {'default', 'mlp', 'cnn', 'cnn105', 'resnet50', 'hrnet_w18', 'vit_tiny_patch16_224', 'cnncompare', 'cnn105compare'} }. You wrote: {config.fsl.model}")