from data_measurements.measurements import TextDuplicates
from datasets import Dataset

dataset = Dataset.from_dict(
    {
        "text": ["Hello", "World", "Hello", "Foo Bar", "wasn", ""],
        "label": [1, 2, 1, 1, 0, 0],
    }
)


def tokenize(sentence: str):
    return sentence.split()


TextDuplicates.standalone(dataset=dataset, feature="text").launch()
