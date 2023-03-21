from typing import List, Dict, Callable
from datasets import Dataset

from data_measurements.measurements.base import DataMeasurement, TokenizedDatasetMixin

from statistics import mean, stdev


class TextLengths(TokenizedDatasetMixin, DataMeasurement):
    name = "text_lengths"

    def __init__(self, tokenizer: Callable[[str], List[str]], *args, **kwargs):
        self.tokenizer = tokenizer
        super().__init__(*args, **kwargs)

    def measure_tokenized(self, dataset: Dataset) -> Dict:
        dataset = dataset.map(lambda x: {**x, "length": len(x["tokenized_text"])})
        df = dataset.to_pandas()
        df["length"] = df.tokenized_text.apply(len)

        avg_length = mean(df.length)
        std_length = stdev(df.length)
        num_uniq_lengths = len(df.length.unique())

        return {
            "average_instance_length": avg_length,
            "standard_dev_instance_length": std_length,
            "num_instance_lengths": num_uniq_lengths,
        }
