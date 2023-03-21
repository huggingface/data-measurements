import pytest


@pytest.fixture
def dummy_tokenizer():
    def tokenize(sentence: str):
        return sentence.split()

    return tokenize
