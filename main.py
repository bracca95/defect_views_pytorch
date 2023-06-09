import os
import sys
import torch
import random
import numpy as np

from src.models.model_utils import Model
from src.train_test.routine import TrainTestExample
from src.datasets.dataset_utils import DatasetBuilder
from src.utils.config_parser import Config
from src.utils.tools import Logger
from config.consts import General as _CG

SEED = 1234         # with the first protonet implementation I used 7

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


if __name__=="__main__":
    try:
        config = Config.deserialize("config/config.json")
    except Exception as e:
        Logger.instance().critical(e.args)
        sys.exit(-1)

    try:
        dataset = DatasetBuilder.load_dataset(config)
    except ValueError as ve:
        Logger.instance().critical(ve.args)
        sys.exit(-1)

    # compute mean and variance of the dataset if not done yet
    if config.dataset_mean is None and config.dataset_std is None:
        Logger.instance().warning("No mean and std set: computing and storing values.")
        DatasetBuilder.compute_mean_std(dataset, config)
        sys.exit(0)

    # instantiate model
    model = Model().to(_CG.DEVICE)
    
    # split dataset
    subsets_dict = DatasetBuilder.split_dataset(dataset, config.dataset_splits)
    
    # train/test
    routine = TrainTestExample(model, dataset, subsets_dict)
    routine.train(config)
    routine.test(config, model_path)
