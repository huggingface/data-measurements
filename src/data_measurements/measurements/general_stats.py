from datasets import Dataset
import pandas as pd
import gradio as gr
import nltk
from nltk.corpus import stopwords

from data_measurements.measurements.base import (
    DataMeasurement,
    DataMeasurementResults,
    TokenizedDatasetMixin,
    Widget
)
from data_measurements.measurements.text_duplicates import TextDuplicates


import utils

logs = utils.prepare_logging(__file__)

CNT = "count"
VOCAB = "vocab"
PROP = "proportion"
# TODO: Read this in depending on chosen language / expand beyond english
nltk.download("stopwords", quiet=True)
_CLOSED_CLASS = (
        stopwords.words("english")
        + ["t", "n", "ll", "d", "s"]
        + ["wasn", "weren", "won", "aren", "wouldn", "shouldn", "didn", "don",
           "hasn", "ain", "couldn", "doesn", "hadn", "haven", "isn", "mightn",
           "mustn", "needn", "shan", "would", "could", "dont"]
        + [str(i) for i in range(0, 99)]
)
_TOP_N = 100


class GeneralStatsResults(DataMeasurementResults):
    def __init__(
            self,
            total_words,
            total_open_words,
            sorted_top_vocab_df,
            text_nan_count,
            dups_frac,
    ):
        self.total_words = total_words
        self.total_open_words = total_open_words
        self.sorted_top_vocab_df = sorted_top_vocab_df
        self.text_nan_count = text_nan_count
        self.dups_frac = dups_frac

    def __eq__(self, other):
        pass

    def to_figure(self):
        pass


class GeneralStatsWidget(Widget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.general_stats = gr.Markdown(render=False)
        self.general_stats_top_vocab = gr.DataFrame(render=False)
        self.general_stats_missing = gr.Markdown(render=False)
        self.general_stats_duplicates = gr.Markdown(render=False)

    def render(self):
        with gr.TabItem(f"General Text Statistics"):
            self.general_stats.render()
            self.general_stats_missing.render()
            self.general_stats_duplicates.render()
            self.general_stats_top_vocab.render()

    def update(self, results: GeneralStatsResults):
        general_stats_text = f"""
        Use this widget to check whether the terms you see most represented in the dataset make sense for the goals of the dataset.

        There are {str(results.total_words)} total words.

        There are {results.total_open_words} after removing closed class words.

        The most common [open class words](https://dictionary.apa.org/open-class-words) and their counts are: 
        """

        top_vocab = pd.DataFrame(results.sorted_top_vocab_df).round(4)

        missing_text = (
            f"There are {results.text_nan_count} missing values in the dataset"
        )

        if results.dups_frac > 0:
            dupes_text = f"The dataset is {round(results.dups_frac * 100, 2)}% duplicates, For more information about the duplicates, click the 'Duplicates' tab."
        else:
            dupes_text = "There are 0 duplicate items in the dataset"

        return {
            self.general_stats: general_stats_text,
            self.general_stats_top_vocab: top_vocab,
            self.general_stats_missing: missing_text,
            self.general_stats_duplicates: dupes_text,
        }

    @property
    def output_components(self):
        return [
            self.general_stats,
            self.general_stats_top_vocab,
            self.general_stats_missing,
            self.general_stats_duplicates,
        ]

    def add_events(self, state: gr.State):
        pass


def count_vocab_frequencies(dataset: Dataset):
    return (
        pd.DataFrame({"tokenized": dataset["tokenized_text"]})
        .tokenized.explode()
        .value_counts()
        .to_frame(name="count")
    )


def calc_p_word(word_count_df):
    word_count_df[PROP] = word_count_df[CNT] / float(sum(word_count_df[CNT]))
    vocab_counts_df = pd.DataFrame(
        word_count_df.sort_values(by=CNT, ascending=False))
    vocab_counts_df[VOCAB] = vocab_counts_df.index
    return vocab_counts_df


def filter_vocab(vocab_counts_df):
    # TODO: Add warnings (which words are missing) to log file?
    filtered_vocab_counts_df = vocab_counts_df.drop(_CLOSED_CLASS, errors="ignore")
    filtered_count = filtered_vocab_counts_df[CNT]
    filtered_count_denom = float(sum(filtered_vocab_counts_df[CNT]))
    filtered_vocab_counts_df[PROP] = filtered_count / filtered_count_denom
    return filtered_vocab_counts_df


class GeneralStats(TokenizedDatasetMixin, DataMeasurement):
    name = "general_stats"
    widget = GeneralStatsWidget

    def measure(self, dataset: Dataset) -> GeneralStatsResults:
        dataset = self.tokenize_dataset(dataset)
        word_count_df = count_vocab_frequencies(dataset)
        vocab_counts_df = calc_p_word(word_count_df)

        total_words = len(vocab_counts_df)
        vocab_counts_filtered_df = filter_vocab(vocab_counts_df)
        total_open_words = len(vocab_counts_filtered_df)
        sorted_top_vocab_df = vocab_counts_filtered_df.sort_values(
            "count", ascending=False
        ).head(_TOP_N)

        text_nan_count = int(pd.DataFrame({"tokenized": dataset["tokenized_text"]}).isnull().sum().sum())

        dups_frac = TextDuplicates(feature=self.feature).measure(dataset).duplicate_fraction

        return GeneralStatsResults(
            total_words=total_words,
            total_open_words=total_open_words,
            sorted_top_vocab_df=sorted_top_vocab_df,
            text_nan_count=text_nan_count,
            dups_frac=dups_frac,
        )
