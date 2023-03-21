from typing import Dict
from datasets import Dataset

from data_measurements.measurements.base import DataMeasurement, EvaluateMixin


class TextDuplicates(EvaluateMixin, DataMeasurement):
    name = "text_duplicates"

    def measure(self, dataset: Dataset) -> Dict:
        return self.metric.compute(data=dataset[self.feature])["duplicate_fraction"]
