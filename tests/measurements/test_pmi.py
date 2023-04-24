import pytest
from datasets import Dataset

from data_measurements.measurements import PMI


@pytest.fixture
def dataset():
    return Dataset.from_list(
        [
            {"text": "he went to the park"},
            {"text": "she has a cat"},
            {"text": "their car is blue"},
        ]
    )


def test_pmi_initialize(dummy_tokenizer):
    PMI(tokenizer=dummy_tokenizer, feature=None)


def test_pmi_run(dummy_tokenizer, dataset):
    pmi = PMI(tokenizer=dummy_tokenizer, feature="text")
    pmi.measure(dataset)
