from data_measurements import DataMeasurementSuite
from data_measurements.measurements import TextDuplicates


def test_end_to_end():
    suite = DataMeasurementSuite(
        dataset="society-ethics/data-measurements-end-to-end-test",
        measurements=[
            TextDuplicates,
        ],
        feature="text",
        split="train",
    )

    results = suite.run()

    assert results["text_duplicates"] == 0.25
