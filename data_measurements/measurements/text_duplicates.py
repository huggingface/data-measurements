from typing import List, Dict

from data_measurements.measurements.base import DataMeasurement


class TextDuplicates(DataMeasurement):
    def __init__(self):
        super().__init__(metric_name="text_duplicates")

    def measure(self, dataset: List[str]) -> Dict:
        return self.metric.compute(data=dataset)["duplicate_fraction"]
