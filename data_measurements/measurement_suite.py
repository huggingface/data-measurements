from typing import List, Type, Dict

from datasets import load_dataset
from data_measurements.measurements import DataMeasurement


class DataMeasurementSuite:
    def __init__(
            self,
            dataset: str,
            feature: str,
            split: str,
            measurements: List[Type[DataMeasurement]],
    ):
        self.dataset: List[str] = load_dataset(dataset, split=split)[feature]
        self.measurements = [DataMeasurement.create(m) for m in measurements]

    def run(self) -> Dict:
        return {m.name: m.measure(dataset=self.dataset) for m in self.measurements}
