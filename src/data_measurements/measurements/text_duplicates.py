from datasets import Dataset
import utils.dataset_utils as ds_utils
import gradio as gr

from typing import Dict

from data_measurements.measurements.base import DataMeasurement, DataMeasurementResults, EvaluateMixin, Widget


class TextDuplicatesResults(DataMeasurementResults):
    def __init__(
            self,
            duplicate_fraction: float,
            duplicates_dict: Dict,
    ):
        self.duplicate_fraction = duplicate_fraction
        self.duplicates_dict = duplicates_dict

    def __eq__(self, other):
        if isinstance(other, TextDuplicatesResults):
            try:
                assert self.duplicate_fraction == other.duplicate_fraction
                return True
            except AssertionError:
                return False
        else:
            return False

    def to_figure(self):
        pass


class TextDuplicatesWidget(Widget):
    def __init__(self):
        duplicates_text = f"""
        Use this widget to identify text strings that appear more than once. 

        A model's training and testing may be negatively affected by unwarranted duplicates ([Lee et al., 2021](https://arxiv.org/abs/2107.06499))

        ------

        ### Here is the list of all the duplicated items and their counts in the dataset. 
        """
        self.duplicates_intro = gr.Markdown(render=False, value=duplicates_text)
        self.duplicates_df = gr.DataFrame(render=False)
        self.duplicates_text = gr.Markdown(render=False)

    def render(self):
        with gr.TabItem(f"Duplicates"):
            self.duplicates_intro.render()
            self.duplicates_text.render()
            self.duplicates_df.render()

    def update(self, results: TextDuplicatesResults):
        output = {}

        if not results.duplicates_dict:
            output[self.duplicates_df] = gr.DataFrame.update(visible=False)
            output[self.duplicates_text] = gr.Markdown.update(visible=True,
                                                              value="There are no duplicates in this dataset! 🥳")
        else:
            dupes_df_tmp = ds_utils.counter_dict_to_df(results.duplicates_dict, key_as_column=True)
            dupes_df_tmp.columns = ["instance", "count"]
            # Nice to have the counts show up first, because the instances
            # can be quite long (and run off the page)
            dupes_df = dupes_df_tmp[["count", "instance"]]
            output[self.duplicates_df] = gr.DataFrame.update(visible=True, value=dupes_df)

            duplicates_text = f"The fraction of data that is duplicate is {str(round(results.duplicate_fraction, 4))}"
            output[self.duplicates_text] = gr.Markdown.update(value=duplicates_text, visible=True)

        return output

    @property
    def output_components(self):
        return [
            self.duplicates_text,
            self.duplicates_df,
        ]

    def add_events(self, state: gr.State):
        pass


class TextDuplicates(EvaluateMixin, DataMeasurement):
    name = "text_duplicates"
    widget = TextDuplicatesWidget

    def measure(self, dataset: Dataset) -> TextDuplicatesResults:
        # TODO: list_duplicates is memory-intensive for large datasets
        # TODO: Consider making that an option, not the default.
        metric_output = super().run_metric(dataset, list_duplicates=True)

        return TextDuplicatesResults(
            duplicate_fraction=metric_output["duplicate_fraction"],
            duplicates_dict=metric_output.get("duplicates_dict", {}),
        )
