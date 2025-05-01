from collections import OrderedDict

import torch
import tqdm

from restobox.metrics.metric import Metric
from restobox.reporting.report_writer import ReportWriter
from restobox.reporting.visdom_report_writer import VisdomReportWriter


def get_metric_value(metric: Metric, value: float) -> float | str:
    if metric.format_string is not None:
        return format(value, metric.format_string)
    return value


class ProgressReportWriter(ReportWriter):
    def __init__(self,target: tqdm.tqdm) -> None:
        super().__init__()
        self.target = target

    def update(self,
               epoch: int,
               step_in_epoch: int,
               global_step: int,
               metrics: list[Metric]) -> None:
        metrics_to_write = OrderedDict()


        for metric in metrics:
            metrics_to_write["epoch_" + metric.name] = get_metric_value(metric,metric.epoch_value)
            metrics_to_write["current_" + metric.name ] = get_metric_value(metric,metric.report_value)

        self.target.set_postfix(metrics_to_write)

    def update_images(self,
                      epoch: int,
                      step_in_epoch: int,
                      global_step: int,
                      images: dict[str,torch.Tensor]) -> None:
        pass

