from torch import nn

from src.models.model import Model
from src.train_test.routine import TrainTest
from src.train_test.standard_routine import StandardRoutine
from src.train_test.proto_routine import ProtoRoutine
from src.datasets.defectviews import GlassOpt
from config.consts import SubsetsDict

class RoutineBuilder:

    @staticmethod
    def build_routine(name: str, model: Model, dataset: GlassOpt, subsets_dict: SubsetsDict) -> TrainTest:
        if "compare" in name:
            return StandardRoutine(model, dataset, subsets_dict)
        else:
            return ProtoRoutine(model, dataset, subsets_dict)