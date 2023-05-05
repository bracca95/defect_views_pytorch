import torch

from abc import ABC, abstractmethod
from torch import nn
from typing import Optional
from src.datasets.defectviews import DefectViews

from src.utils.tools import Logger
from src.utils.config_parser import Config
from src.datasets.defectviews import DefectViews
from src.datasets.dataset_utils import SubsetInfo
from config.consts import SubsetsDict, General


class TrainTest(ABC):

    train_str, val_str, test_str = General.DEFAULT_SUBSETS
    
    def __init__(self, model: nn.Module, dataset: DefectViews, subsets_dict: Optional[SubsetsDict]=None):
        self.model = model
        self.dataset = dataset
        self.subsets_dict = subsets_dict

        self.train_info: Optional[SubsetInfo] = self.get_subset_info(self.train_str)
        self.val_info: Optional[SubsetInfo] = self.get_subset_info(self.val_str)
        self.test_info: Optional[SubsetInfo] = self.get_subset_info(self.test_str)

    @abstractmethod
    def train(self, config: Config):
        ...

    @abstractmethod
    def test(self, config: Config):
        ...

    def get_subset_info(self, subset_str_id: str) -> Optional[SubsetInfo]:
        """Wrap subset into SubsetInfo structure (holds more information)

        Args:
            subset_str_id (str): { 'train', 'val', 'test' }

        Returns:
            SubsetInfo if the Subset is present (validation dataset can be None)

        Raises:
            ValueError if `subset_str_id` is not in { 'train', 'val', 'test' }
        """

        if subset_str_id not in General.DEFAULT_SUBSETS:
            raise ValueError(f"TrainTest::get_subset_info: only accept 'train', 'val', 'test'")
        
        if self.subsets_dict is None:
            return None
        
        if self.subsets_dict[subset_str_id] is None:
            info_dict = None
        else:
            subset_labels = [self.dataset[idx][1] for idx in self.subsets_dict[subset_str_id].indices]
            classes = list(set(subset_labels))
            info_dict = { self.dataset.idx_to_label[i]: subset_labels.count(i) for i in classes }
            Logger.instance().debug(f"{subset_str_id} has {len(classes)} classes: {info_dict}")
        
        return SubsetInfo(subset_str_id, self.subsets_dict[subset_str_id], info_dict)


class TrainTestExample(TrainTest):

    def __init__(self, model: nn.Module, dataset: DefectViews, subsets_dict: Optional[SubsetsDict]=None):
        super().__init__(model, dataset, subsets_dict)

    def train(self):
        Logger.instance().debug("train example")

    def test(self):
        Logger.instance().debug("test example")