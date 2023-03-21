from unittest.mock import MagicMock

from data_measurements import DataMeasurementSuite
from data_measurements.measurements import TextDuplicates


def test_measurement_suite_initialize(monkeypatch):
    load_dataset = MagicMock()
    monkeypatch.setattr("data_measurements.measurement_suite.load_dataset", load_dataset)
    mock_metric = MagicMock()
    load_metric = MagicMock(lambda x: mock_metric)
    monkeypatch.setattr("data_measurements.measurements.base.load_metric", load_metric)

    suite = DataMeasurementSuite(
        dataset="imdb",
        measurements=[
            TextDuplicates,
        ],
        feature="text",
        split="train",
    )

    load_dataset.assert_called_with("imdb", split="train")

    assert len(suite.measurements) == 1
