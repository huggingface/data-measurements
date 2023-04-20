from data_measurements import DataMeasurementSuite
from data_measurements.measurements import (
    Cooccurences
)
from sklearn.feature_extraction.text import CountVectorizer


tokenizer = CountVectorizer(token_pattern="(?u)\\b\\w+\\b").build_tokenizer()

import time
t = time.time()

suite = DataMeasurementSuite(
    dataset="hate_speech18",
    feature="text",
    label="label",
    split="train",
    tokenizer=tokenizer,
    measurements=[
        Cooccurences
    ],
)

results = suite.run()

elapsed = time.time() - t
print(elapsed)
# results["cooccurences"]
