from unittest.mock import MagicMock

import pytest

from data_measurements.measurements import TextDuplicates


@pytest.fixture
def mock_load_metric(monkeypatch):
    mock_metric = MagicMock()
    load_metric = MagicMock(lambda x: mock_metric)

    monkeypatch.setattr("data_measurements.measurements.base.load_metric", load_metric)

    return load_metric


def test_text_duplicates_initialize(mock_load_metric):
    TextDuplicates()
    mock_load_metric.assert_called_once_with("text_duplicates")


def test_text_duplicates_run(mock_load_metric):
    dataset = ["Hello", "World", "Hello", "Foo Bar"]
    text_duplicates = TextDuplicates()
    text_duplicates.measure(dataset)

    text_duplicates.metric.compute.assert_called_once_with(data=dataset)
