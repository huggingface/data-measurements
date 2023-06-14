from data_measurements.measurements import TextLengths
from datasets import Dataset

# TODO: The dataset and the loading args/kwargs for the measurement should probably be passed in
dataset = Dataset.from_dict(
    {
        "text": ["Hello", "World", "Hello", "Foo Bar", "wasn", ""],
        "label": [1, 2, 1, 1, 0, 0],
    }
)


def tokenize(sentence: str):
    return sentence.split()


TextLengths.standalone(dataset=dataset, feature="text", tokenizer=tokenize).launch()
