from statistics import mean, stdev

import gradio as gr

import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
from pandas import DataFrame

from data_measurements.measurements.base import (
    DataMeasurement,
    DataMeasurementResults,
    TokenizedDatasetMixin,
    Widget
)


class TextLengthsResults(DataMeasurementResults):
    def __init__(
        self,
        average_instance_length: float,
        standard_dev_instance_length: float,
        num_instance_lengths: int,
        lengths: DataFrame,
    ):
        super().__init__()
        self.average_instance_length = average_instance_length
        self.standard_dev_instance_length = standard_dev_instance_length
        self.num_instance_lengths = num_instance_lengths
        self.lengths = lengths

    def __eq__(self, other):
        if isinstance(other, TextLengthsResults):
            try:
                assert self.average_instance_length == other.average_instance_length
                assert self.standard_dev_instance_length == other.standard_dev_instance_length
                assert self.num_instance_lengths == other.num_instance_lengths
                assert all(self.lengths == other.lengths)
                return True
            except AssertionError:
                return False
        else:
            return False

    def to_figure(self):
        # TODO: Copy and pasted... clean it and test
        # TODO: Write it OOP-style if possible (see the matplotlib guide)
        fig, axs = plt.subplots(figsize=(15, 6), dpi=150)
        plt.xlabel("Number of tokens")
        plt.title("Binned counts of text lengths, with kernel density estimate and ticks for each instance.")
        sns.histplot(data=self.lengths, kde=True, ax=axs, legend=False)
        sns.rugplot(data=self.lengths, ax=axs)
        return fig


def update_text_length_df(length, results: TextLengthsResults):
    return DataFrame(results.lengths[results.lengths == length])


class TextLengthsWidget(Widget):
    def __init__(self):
        self.text_length_distribution_plot = gr.Plot(render=False)
        self.text_length_explainer = gr.Markdown(render=False)
        self.text_length_drop_down = gr.Dropdown(render=False)
        self.text_length_df = gr.DataFrame(render=False)

    def render(self):
        with gr.TabItem("Text Lengths"):
            gr.Markdown(
                "Use this widget to identify outliers, particularly suspiciously long outliers."
            )
            gr.Markdown(
                "Below, you can see how the lengths of the text instances in your "
                "dataset are distributed."
            )
            gr.Markdown(
                "Any unexpected peaks or valleys in the distribution may help to "
                "identify instances you want to remove or augment."
            )
            gr.Markdown(
                "### Here is the count of different text lengths in " "your dataset:"
            )
            # When matplotlib first creates this, it's a Figure.
            # Once it's saved, then read back in,
            # it's an ndarray that must be displayed using st.image
            # (I know, lame).
            self.text_length_distribution_plot.render()
            self.text_length_explainer.render()
            self.text_length_drop_down.render()
            self.text_length_df.render()

    def update(self, results: TextLengthsResults):
        explainer_text = (
            "The average length of text instances is **"
            + str(round(results.average_instance_length, 2))
            + " words**, with a standard deviation of **"
            + str(round(results.standard_dev_instance_length, 2))
            + "**."
        )
        # TODO: Add text on choosing the length you want to the dropdown.
        output = {
            self.text_length_distribution_plot: results.to_figure(),
            self.text_length_explainer: explainer_text,
        }
        if results.lengths is not None:
            import numpy as np

            choices = np.sort(results.lengths.unique())[
                ::-1
            ].tolist()
            output[self.text_length_drop_down] = gr.Dropdown.update(
                choices=choices, value=choices[0]
            )
            output[self.text_length_df] = update_text_length_df(choices[0], results)
        else:
            output[self.text_length_df] = gr.update(visible=False)
            output[self.text_length_drop_down] = gr.update(visible=False)
        return output

    @property
    def output_components(self):
        return [
            self.text_length_distribution_plot,
            self.text_length_explainer,
            self.text_length_drop_down,
            self.text_length_df,
        ]

    def add_events(self, state: gr.State):
        self.text_length_drop_down.change(
            update_text_length_df,
            inputs=[self.text_length_drop_down, state],
            outputs=[self.text_length_df],
        )


class TextLengths(TokenizedDatasetMixin, DataMeasurement):
    name = "text_lengths"
    widget = TextLengthsWidget

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def measure(self, dataset: Dataset) -> TextLengthsResults:
        # TODO: See if it's possible to do the tokenization with a decorator or something...
        dataset = self.tokenize_dataset(dataset)
        dataset = dataset.map(lambda x: {**x, "length": len(x["tokenized_text"])})
        df = dataset.to_pandas()
        df["length"] = df.tokenized_text.apply(len)

        avg_length = mean(df.length)
        std_length = stdev(df.length)
        num_uniq_lengths = len(df.length.unique())

        return TextLengthsResults(
            average_instance_length=avg_length,
            standard_dev_instance_length=std_length,
            num_instance_lengths=num_uniq_lengths,
            lengths=df.length,
        )
