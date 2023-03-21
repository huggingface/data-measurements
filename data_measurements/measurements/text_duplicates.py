from data_measurements.measurements.base import DataMeasurement, EvaluateMixin


class TextDuplicates(EvaluateMixin, DataMeasurement):
    name = "text_duplicates"
