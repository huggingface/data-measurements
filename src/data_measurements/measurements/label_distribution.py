from typing import Dict, List

import plotly.express as px
from datasets import Dataset
import gradio as gr

from data_measurements.measurements.base import (
    DataMeasurement,
    DataMeasurementResults,
    EvaluateMixin,
    LabelMeasurementMixin,
    Widget
)

import utils

logs = utils.prepare_logging(__file__)


class LabelDistributionResults(DataMeasurementResults):
    def __init__(self, label_distribution: Dict[str, List[float]], label_skew: float):
        self.label_distribution = label_distribution
        self.label_skew = label_skew

    def __eq__(self, other):
        if isinstance(other, LabelDistributionResults):
            try:
                assert self.label_distribution == other.label_distribution
                assert self.label_skew == other.label_skew
                return True
            except AssertionError:
                return False
        else:
            return False

    def to_figure(self):
        fig_labels = px.pie(
            names=self.label_distribution["labels"],
            values=self.label_distribution["fractions"],
        )
        return fig_labels


class LabelDistributionWidget(Widget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_dist_plot = gr.Plot(render=False, visible=False)
        self.label_dist_no_label_text = gr.Markdown(
            value="No labels were found in the dataset", render=False, visible=False
        )
        self.label_dist_accordion = gr.Accordion(render=False, label="", open=False)

    def render(self):
        with gr.TabItem(label="Label Distribution"):
            gr.Markdown(
                "Use this widget to see how balanced the labels in your dataset are."
            )
            self.label_dist_plot.render()
            self.label_dist_no_label_text.render()

    def update(self, results: LabelDistributionResults):
        output = {
            self.label_dist_plot: gr.Plot.update(
                value=results.to_figure(), visible=True
            ),
            self.label_dist_no_label_text: gr.Markdown.update(visible=False),
        }
        return output

    @property
    def output_components(self):
        return [self.label_dist_plot, self.label_dist_no_label_text]

    def add_events(self, state: gr.State):
        pass


class LabelDistribution(LabelMeasurementMixin, EvaluateMixin, DataMeasurement):
    name = "label_distribution"
    widget = LabelDistributionWidget

    def measure(self, dataset: Dataset) -> LabelDistributionResults:
        results = super().run_metric(dataset)
        return LabelDistributionResults(
            label_distribution=results["label_distribution"],
            label_skew=results["label_skew"],
        )
