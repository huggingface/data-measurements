import abc
from abc import ABC
from typing import Callable, Dict, List, Type

import evaluate
from datasets import Dataset
from evaluate import load as load_metric
import gradio as gr


class DataMeasurementResults(ABC):
    @abc.abstractmethod
    def to_figure(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def __eq__(self, other):
        raise NotImplementedError()


class Widget(ABC):
    @abc.abstractmethod
    def render(self):
        pass

    @abc.abstractmethod
    def update(self, results: DataMeasurementResults):
        pass

    @property
    @abc.abstractmethod
    def output_components(self):
        pass

    @abc.abstractmethod
    def add_events(self, state: gr.State):
        pass


class DataMeasurement(ABC):
    def __init__(self, feature: str, *args, **kwargs):
        self.feature = feature

    @property
    @abc.abstractmethod
    def name(self):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def widget(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def measure(self, dataset) -> DataMeasurementResults:
        raise NotImplementedError()

    @classmethod
    def standalone(cls):
        with gr.Blocks() as demo:
            # TODO: The dataset and the loading args/kwargs for the measurement should probably be passed in
            dataset = Dataset.from_dict(
                {
                    "text": ["Hello", "World", "Hello", "Foo Bar"],
                    "label": [1, 2, 1, 1],
                }
            )
            measurement = cls(feature="label")
            results = measurement.measure(dataset)

            widget = measurement.widget()
            widget.render()

            def update_ui():
                return widget.update(results)

            demo.load(
                update_ui,
                inputs=[],
                outputs=widget.output_components
            )

        return demo


class DataMeasurementFactory:
    @classmethod
    def create(cls, measurement: Type[DataMeasurement], *args, **kwargs):
        arguments = {"feature": kwargs["feature"]}

        if issubclass(measurement, TokenizedDatasetMixin):
            arguments["tokenizer"] = kwargs["tokenizer"]

        if issubclass(measurement, LabelMeasurementMixin):
            arguments["feature"] = kwargs["label"]

        return measurement(**arguments)


class EvaluateMixin:
    name: str
    feature: str

    def __init__(self, *args, **kwargs):
        self.metric: evaluate.EvaluationModule = load_metric(self.name)
        super().__init__(*args, **kwargs)

    def run_metric(self, dataset: Dataset) -> Dict:
        return self.metric.compute(data=dataset[self.feature])


class TokenizedDatasetMixin:
    tokenizer: Callable[[str], List[str]]
    feature: str

    def __init__(self, tokenizer: Callable[[str], List[str]], *args, **kwargs):
        self.tokenizer = tokenizer
        super().__init__(*args, **kwargs)

    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        return dataset.map(lambda x: {**x, "tokenized_text": self.tokenizer(x[self.feature])})


class LabelMeasurementMixin:
    pass
