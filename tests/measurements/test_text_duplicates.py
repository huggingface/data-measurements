from unittest.mock import MagicMock

import pytest

from data_measurements.measurements import TextDuplicates
from datasets import Dataset

mock_result = {"duplicate_fraction": "Mock Result"}


@pytest.fixture
def mock_load_metric(monkeypatch):
    mock_metric = MagicMock()
    mock_metric.compute.return_value = mock_result
    load_metric = MagicMock()
    load_metric.return_value = mock_metric

    monkeypatch.setattr("data_measurements.measurements.base.load_metric", load_metric)

    return load_metric


def test_text_duplicates_initialize(mock_load_metric):
    TextDuplicates(feature=None)
    mock_load_metric.assert_called_once_with("text_duplicates")


def test_text_duplicates_run(mock_load_metric):
    dataset = Dataset.from_dict({
        "text": ["Hello", "World", "Hello", "Foo Bar"]
    })
    text_duplicates = TextDuplicates(feature="text")
    result = text_duplicates.measure(dataset)

    text_duplicates.metric.compute.assert_called_once_with(data=dataset["text"])
    assert result == mock_result["duplicate_fraction"]
