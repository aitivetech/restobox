import numpy as np
import torch
from visdom import Visdom

from restobox.core.tensors import tensor_to_uint8_numpy
from restobox.metrics.metric import Metric
from restobox.reporting.report_writer import ReportWriter


class VisdomReportWriter(ReportWriter):
    def __init__(self, env: str = "main",frequency: int = 100) -> None:
        super().__init__()
        self.viz = Visdom()
        self.env = env
        self.frequency = frequency
        self.value_plots = {}
        self.image_plots = {}

        self.viz.delete_env(env=self.env)

    def update(self,
               epoch: int,
               step_in_epoch: int,
               global_step: int,
               metrics: list[Metric]) -> None:

        if global_step % self.frequency == 0:
            for metric in metrics:
                self._plot("epoch_" + metric.name, global_step, metric.epoch_value)
                self._plot("current_" + metric.name, global_step, metric.report_value)

    def update_images(self,
                      epoch: int,
                      step_in_epoch: int,
                      global_step: int, images: dict[str, torch.Tensor]) -> None:

        images_towrite = {
            k: tensor_to_uint8_numpy(v) for k, v in images.items()
        }

        for k, v in images_towrite.items():
            if k not in self.image_plots:
                self.image_plots[k] = self.viz.images(v, opts=dict(title=k),env=self.env)
            else:
                self.viz.images(v, opts=dict(title=k),env=self.env,win=self.image_plots[k])

    def _plot(self, name: str, global_step: int, value: float) -> None:
        if name not in self.value_plots:
            self.value_plots[name] = self.viz.line(X=np.array([global_step, global_step]), Y=np.array([value, value]),
                                                   env=self.env, opts=dict(
                    legend=[name],
                    title=name,
                    xlabel='Steps',
                    ylabel=name
                ))
        else:
            self.viz.line(X=np.array([global_step]), Y=np.array([value]), env=self.env, win=self.value_plots[name],
                          update='append')
