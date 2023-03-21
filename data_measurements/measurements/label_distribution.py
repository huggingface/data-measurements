from data_measurements.measurements.base import DataMeasurement, EvaluateMixin, LabelMeasurementMixin


class LabelDistribution(LabelMeasurementMixin, EvaluateMixin, DataMeasurement):
    name = "label_distribution"
