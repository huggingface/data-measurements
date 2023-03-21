from data_measurements import DataMeasurementSuite
from data_measurements.measurements import TextDuplicates, TextLengths


def test_end_to_end(dummy_tokenizer):
    suite = DataMeasurementSuite(
        dataset="society-ethics/data-measurements-end-to-end-test",
        feature="text",
        split="train",
        tokenizer=dummy_tokenizer,
        measurements=[
            TextDuplicates,
            TextLengths,
        ],
    )

    results = suite.run()

    assert results["text_duplicates"] == 0.25
    assert results["text_lengths"] == {
        "average_instance_length": 2.25,
        "standard_dev_instance_length": 0.5,
        "num_instance_lengths": 2
    }
