import pytest
from datasets import Dataset

from data_measurements.measurements import Cooccurences


@pytest.fixture
def dataset():
    return Dataset.from_list(
        [
            {"text": "he went to the park"},
            {"text": "she has a cat"},
            {"text": "their car is blue"},
        ]
    )


def test_cooccurences_initialize(dummy_tokenizer):
    Cooccurences(tokenizer=dummy_tokenizer, feature=None)


def test_cooccurences_run(dummy_tokenizer, dataset):
    cooccurences = Cooccurences(tokenizer=dummy_tokenizer, feature="text")
    cooccurences.measure(dataset)
