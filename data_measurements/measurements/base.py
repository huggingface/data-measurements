from typing import Dict, List
import abc
from abc import ABC

import evaluate
from evaluate import load as load_metric


class DataMeasurement(ABC):
    def __init__(self, metric_name: str):
        self.metric: evaluate.EvaluationModule = load_metric(metric_name)
        self.name = metric_name

    @abc.abstractmethod
    def measure(self, dataset: List[str]) -> Dict:
        raise NotImplementedError()

    @classmethod
    def create(cls, measurement):
        return measurement()
