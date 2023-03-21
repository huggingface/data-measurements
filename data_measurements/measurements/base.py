from typing import Dict, List, Callable, Type
import abc
from abc import ABC

import evaluate
from evaluate import load as load_metric
from datasets import Dataset


class DataMeasurement(ABC):
    def __init__(self, feature: str, *args, **kwargs):
        self.feature = feature

    @property
    @abc.abstractmethod
    def name(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def measure(self, dataset: List[str]) -> Dict:
        raise NotImplementedError()


class DataMeasurementFactory():
    @classmethod
    def create(cls, measurement: Type[DataMeasurement], *args, **kwargs):
        arguments = {
            "feature": kwargs["feature"]
        }

        if issubclass(measurement, TokenizedDatasetMixin):
            arguments["tokenizer"] = kwargs["tokenizer"]

        if issubclass(measurement, LabelMeasurementMixin):
            arguments["feature"] = kwargs["label"]

        return measurement(**arguments)


class EvaluateMixin:
    name: str
    feature: str

    def __init__(self, *args, **kwargs):
        self.metric: evaluate.EvaluationModule = load_metric(self.name)
        super().__init__(*args, **kwargs)

    def measure(self, dataset: Dataset):
        return self.metric.compute(data=dataset[self.feature])


class TokenizedDatasetMixin:
    tokenizer: Callable[[str], List[str]]
    feature: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        return dataset.map(lambda x: {**x, "tokenized_text": self.tokenizer(x[self.feature])})

    def measure(self, dataset: Dataset) -> Dict:
        dataset = self.tokenize_dataset(dataset)
        return self.measure_tokenized(dataset)

    @abc.abstractmethod
    def measure_tokenized(self, dataset: Dataset) -> Dict:
        raise NotImplementedError()


class LabelMeasurementMixin:
    pass
