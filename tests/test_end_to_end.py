import pandas as pd

from data_measurements import DataMeasurementSuite
from data_measurements.measurements import (
    LabelDistribution,
    LabelDistributionResults,
    TextDuplicates,
    TextDuplicatesResults,
    TextLengths,
    TextLengthsResults,
)


def test_end_to_end(dummy_tokenizer):
    # TODO: Some datasets for end2end testing: hate_speech18, wiki_qa (has a funky format), c4 (timing tests for the first CHUNK of it (250k lines)) –see _DATASET_LIST and _STREAMABLE_DATASET_LIST...
    suite = DataMeasurementSuite(
        dataset="society-ethics/data-measurements-end-to-end-test",
        feature="text",
        label="label",
        split="train",
        tokenizer=dummy_tokenizer,
        measurements=[
            TextDuplicates,
            TextLengths,
            LabelDistribution,
        ],
    )

    results = suite.run()

    assert results["text_duplicates"] == TextDuplicatesResults(duplicate_fraction=0.25)

    assert results["text_lengths"] == TextLengthsResults(
        average_instance_length=2.25,
        standard_dev_instance_length=0.5,
        num_instance_lengths=2,
        lengths=pd.DataFrame([2, 3, 2, 2])[0],
    )

    assert results["label_distribution"] == LabelDistributionResults(
        label_distribution={"labels": [1, 2], "fractions": [0.75, 0.25]},
        label_skew=1.1547005383792515,
    )
