import pytest
from unittest.mock import MagicMock
from enum import Enum
from typing import Callable
from data_measurements.measurements import (
    TextDuplicates,
    TextLengths,
    LabelDistribution,
)

from datasets import Dataset


@pytest.fixture
def dummy_tokenizer():
    def tokenize(sentence: str):
        return sentence.split()

    return tokenize


@pytest.fixture
def mock_load_metric(monkeypatch):
    load_metric = MagicMock()
    monkeypatch.setattr("data_measurements.measurements.base.load_metric", load_metric)

    return load_metric


@pytest.fixture
def mock_load_dataset(monkeypatch):
    load_dataset = MagicMock()
    monkeypatch.setattr("data_measurements.measurement_suite.load_dataset", load_dataset)

    return load_dataset


class MockMeasureMixin:
    measure: Callable
    name: str

    def __new__(cls, *args, **kwargs):
        mock = MagicMock(spec=cls)
        mock.measure = lambda dataset: cls.measure(self=mock, dataset=dataset)
        mock.name = cls.name
        mock.feature = kwargs["feature"]
        return mock


@pytest.fixture()
def mock_measurements():
    class MockedTextDuplicates(MockMeasureMixin, TextDuplicates):
        def measure(self, dataset: Dataset):
            return {
                "duplicate_fraction": 0.25
            }

    class MockedTextLengths(MockMeasureMixin, TextLengths):
        def measure(self, dataset: Dataset):
            return {
                "average_instance_length": 2.25,
                "standard_dev_instance_length": 0.5,
                "num_instance_lengths": 2
            }

    class MockedLabelDistribution(MockMeasureMixin, LabelDistribution):
        def measure(self, dataset: Dataset):
            return {
                "label_distribution": {
                    "labels": [1, 0, 2],
                    "fractions": [0.1, 0.6, 0.3]
                },
                "label_skew": 0.5
            }

    class Measurements(Enum):
        TextDuplicates = MockedTextDuplicates
        TextLengths = MockedTextLengths
        LabelDistribution = MockedLabelDistribution

    return Measurements
