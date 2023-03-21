from typing import List, Type, Dict, Callable, Optional

from datasets import load_dataset, Dataset
from data_measurements.measurements import DataMeasurement, DataMeasurementFactory


class DataMeasurementSuite:
    def __init__(
            self,
            dataset: str,
            feature: str,
            split: str,
            measurements: List[Type[DataMeasurement]],
            tokenizer: Optional[Callable[[str], List[str]]] = None,
    ):
        self.dataset: Dataset = load_dataset(dataset, split=split)
        self.measurements = [
            DataMeasurementFactory.create(m, tokenizer=tokenizer, feature=feature) for m in measurements
        ]

    def run(self) -> Dict:
        return {m.name: m.measure(dataset=self.dataset) for m in self.measurements}
