import abc
from abc import ABC
from typing import Any

import torch

from restobox.metrics.metric import Metric


class ReportWriter(abc.ABC):

    @abc.abstractmethod
    def update(self,
               epoch: int,
               step_in_epoch: int,
               global_step: int,
               metrics: list[Metric]) -> None:
        pass

    @abc.abstractmethod
    def update_images(self,
                      epoch: int,
                      step_in_epoch: int,
                      global_step: int,
                      images: dict[str, torch.Tensor]) -> None:
        pass


class ChainedReportWriter(ReportWriter):
    def __init__(self, writers: list[ReportWriter]) -> None:
        self.writers = writers

    def update(self,
               epoch: int,
               step_in_epoch: int,
               global_step: int,
               metrics: list[Metric]) -> None:
        for writer in self.writers:
            writer.update(epoch,step_in_epoch,global_step,metrics)

    def update_images(self,
                      epoch: int,
                      step_in_epoch: int,
                      global_step: int,
                      images: dict[str, torch.Tensor]) -> None:
        for writer in self.writers:
            writer.update_images(epoch,step_in_epoch,global_step,images)
