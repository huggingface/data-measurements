import numpy.typing as np
import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import MultiLabelBinarizer

from data_measurements.measurements.base import DataMeasurement, DataMeasurementResults, TokenizedDatasetMixin


def count_vocab_frequencies(dataset: Dataset):
    return (
        pd.DataFrame({"tokenized": dataset["tokenized_text"]})
        .tokenized.explode()
        .value_counts()
        .to_frame(name="count")
    )


def count_words_per_sentence(dataset, vocabulary) -> np.NDArray:
    mlb = MultiLabelBinarizer(classes=vocabulary)
    return mlb.fit_transform(dataset["tokenized_text"])


class CooccurencesResults(DataMeasurementResults):
    def __init__(self, matrix: pd.DataFrame):
        self.matrix = matrix

    def __eq__(self, other):
        if isinstance(other, CooccurencesResults):
            try:
                return True
            except AssertionError:
                return False
        else:
            return False

    def to_figure(self):
        pass


class Cooccurences(TokenizedDatasetMixin, DataMeasurement):
    # TODO: Closed Class words should be included...

    name = "cooccurences"
    identity_terms = [
        "man",
        "woman",
        "non-binary",
        "gay",
        "lesbian",
        "queer",
        "trans",
        "straight",
        "cis",
        "she",
        "her",
        "hers",
        "he",
        "him",
        "his",
        "they",
        "them",
        "their",
        "theirs",
        "himself",
        "herself",
    ]
    # TODO: Locked at 1 right now, make this parameterized?
    min_count = 1
    # min_count = 10

    def measure(self, dataset: Dataset) -> CooccurencesResults:
        dataset = self.tokenize_dataset(dataset)
        word_count_df = count_vocab_frequencies(dataset)
        vocabulary = word_count_df.index
        word_counts_per_sentence = count_words_per_sentence(dataset, vocabulary)

        present_terms = vocabulary.intersection(self.identity_terms)
        min_count = word_count_df.loc[present_terms] >= self.min_count
        present_terms = min_count.loc[min_count["count"]].index

        subgroup = pd.DataFrame(word_counts_per_sentence).T.set_index(vocabulary).loc[present_terms].T
        matrix = pd.DataFrame(word_counts_per_sentence.T.dot(subgroup))

        matrix.columns = present_terms
        matrix.index = vocabulary

        return CooccurencesResults(matrix=matrix)
