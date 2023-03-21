import pytest

from data_measurements import DataMeasurementSuite


@pytest.fixture
def measurements(mock_measurements):
    return [
        mock_measurements.TextDuplicates.value,
        mock_measurements.TextLengths.value,
        mock_measurements.LabelDistribution.value,
    ]


@pytest.fixture
def suite(measurements, mock_load_dataset, mock_load_metric, monkeypatch):
    return DataMeasurementSuite(
        dataset="imdb",
        measurements=measurements,
        feature="text",
        label="label",
        split="train",
        tokenizer=lambda x: x,
    )


def test_measurement_suite_initialize(suite, mock_load_dataset, measurements, monkeypatch):
    mock_load_dataset.assert_called_with("imdb", split="train")

    assert len(suite.measurements) == len(measurements)


def test_measurement_suite_run(suite, measurements, monkeypatch):
    assert suite.measurements[0].feature == "text"
    assert suite.measurements[1].feature == "text"
    assert suite.measurements[2].feature == "label"

    results = suite.run()
    assert results == {
        "text_duplicates": {
            "duplicate_fraction": 0.25
        },
        "text_lengths": {
            "average_instance_length": 2.25,
            "standard_dev_instance_length": 0.5,
            "num_instance_lengths": 2
        },
        "label_distribution": {
            "label_distribution": {
                "labels": [1, 0, 2],
                "fractions": [0.1, 0.6, 0.3]
            },
            "label_skew": 0.5
        }
    }
