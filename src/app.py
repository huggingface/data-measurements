# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import ast
import gradio as gr
import utils
from data_measurements import DataMeasurementSuite
from data_measurements.measurements import (
    GeneralStats,
    LabelDistribution,
    TextLengths,
    TextDuplicates,
)

logs = utils.prepare_logging(__file__)

def get_suite():
    def dummy_tokenizer(sentence: str):
        return sentence.split()

    suite = DataMeasurementSuite(
        dataset="society-ethics/data-measurements-end-to-end-test",
        feature="text",
        label="label",
        split="train",
        tokenizer=dummy_tokenizer,
        measurements=[
            GeneralStats,
            LabelDistribution,
            TextLengths,
            TextDuplicates,
        ],
    )

    return suite

# def get_ui_widgets():
#     """Get the widgets that will be displayed in the UI."""
#     return [
#         # widgets.DatasetDescription(DATASET_NAME_TO_DICT),
#         # widgets.Npmi(),
#         # widgets.Zipf()
#     ]


def get_title(dstats):
    title_str = f"### Showing: {dstats.dset_name} - {dstats.dset_config} - {dstats.split_name} - {'-'.join(dstats.text_field)}"
    logs.info("showing header")
    return title_str


suite = get_suite()

# TODO: I want to run this somewhere smart, I guess? But here is fine for now.
results = suite.run()


def create_demo():
    with gr.Blocks() as demo:
        state = gr.State()
        with gr.Row():
            with gr.Column(scale=1):
                widget_list = [w() for w in suite.widgets]
            with gr.Column(scale=4):
                gr.Markdown("# Data Measurements Tool")
                title = gr.Markdown()
                for widget in widget_list:
                    widget.render()

            def update_ui():
                output = {
                    title: "Temp Title",
                    state: {}  # TODO: Do we even need to store the state as a Gradio object? I don't think so..
                }

                for widget, result in zip(widget_list, results.values()):
                    output.update(widget.update(result))

                return output

            measurements = [comp for output in widget_list for comp in output.output_components]
            demo.load(
                update_ui,
                inputs=[
                    # TODO
                    # dataset_args["dset_name"],
                    # dataset_args["dset_config"],
                    # dataset_args["split_name"],
                    # dataset_args["text_field"]
                ],
                outputs=[title, state] + measurements
            )

    return demo


if __name__ == "__main__":
    create_demo().launch()

# TODO: TEMPORARY
demo = create_demo()
demo.launch()
