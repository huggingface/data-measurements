from typing import Callable, Dict, List, Optional, Type

from datasets import Dataset, load_dataset

from data_measurements.measurements import (
    DataMeasurement,
    DataMeasurementFactory,
    DataMeasurementResults,
)
from data_measurements.measurements.base import Widget


class DataMeasurementSuite:
    def __init__(
        self,
        dataset: str,
        feature: str,
        split: str,
        measurements: List[Type[DataMeasurement]],
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        label: Optional[str] = None,
    ):
        # TODO: TEMPORARY
        self.dataset: Dataset = load_dataset(dataset, split=split)
        self.measurements = [
            DataMeasurementFactory.create(m, tokenizer=tokenizer, feature=feature, label=label) for m in measurements
        ]

    def run(self) -> Dict[str, DataMeasurementResults]:
        return {m.name: m.measure(dataset=self.dataset) for m in self.measurements}

    @property
    def widgets(self) -> List[Widget]:
        return [m.widget for m in self.measurements]
