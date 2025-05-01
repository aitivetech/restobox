from restobox.metrics.metric import Metric


class ExternalMetric(Metric):
    def __init__(self,name: str,format_string: str | None = None):
        super().__init__(name,format_string)

    def set_current_value(self, value:float):
        self.update_value(value)