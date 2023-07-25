from torch import nn

from src.models.model import Model
from src.models.MLP.mlp_basic import MLP
from src.models.CNN.cnn_basic import CNN
from src.models.CNN.cnn_105 import CNN105
from src.models.FSL.ProtoNet.protonet import ProtoNet
from src.models.pretrained.extractors import TimmFeatureExtractor
from src.models.pretrained.classification_head import Head
from src.utils.config_parser import Config
    

class ModelBuilder:

    @staticmethod
    def load_model(config: Config, out_classes) -> Model:
        if config.fsl.model.lower() == "default":
            return ProtoNet(config)
        elif config.fsl.model.lower() == "mlp":
            return MLP(config, config.image_size * config.image_size)
        elif config.fsl.model.lower() == "cnn":
            return CNN(config)
        elif config.fsl.model.lower() == "cnn105":
            return CNN105(config)
        elif config.fsl.model.lower() in ("resnet50", "hrnet_w18", "vit_tiny_patch16_224"):
            return TimmFeatureExtractor(config, config.fsl.model.lower(), in_chans=1, pooled=True, mean=config.dataset_mean, std=config.dataset_std)
        elif config.fsl.model.lower() == "cnncompare":
            extractor = CNN(config)
            return Head(config, extractor, out_classes)
        elif config.fsl.model.lower() == "cnn105compare":
            extractor = CNN105(config)
            return Head(config, extractor, out_classes)
        elif config.fsl.model.lower() in ("resnet50compare", "hrnet_w18compare", "vit_tiny_patch16_224compare"):
            model_name = config.fsl.model.lower().split("compare")[0]
            extractor = TimmFeatureExtractor(config, model_name, in_chans=1, pooled=True, mean=config.dataset_mean, std=config.dataset_std)
            return Head(config, extractor, out_classes)
        else:
            raise ValueError(
                f"fsl.model must be" +
                f"{ {'default', 'mlp', 'cnn', 'cnn105', 'resnet50', 'hrnet_w18', 'vit_tiny_patch16_224'} }" +
                f"or compare variants . You wrote: {config.fsl.model}"
            )