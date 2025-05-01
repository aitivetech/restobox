from abc import ABC, abstractmethod

import torch


class Metric(ABC):
    def __init__(self, name: str,format_string: str | None = None):
        self.name = name
        self.epoch_steps = 0
        self.report_steps = 0
        self.epoch_value_accum = 0
        self.report_value_accum = 0
        self.format_string = format_string

    @property
    def epoch_value(self) -> float:
        return self.epoch_value_accum / self.epoch_steps

    @property
    def report_value(self) -> float:
        return self.report_value_accum / self.report_steps

    def reset(self, reset_epoch: bool,reset_report: bool):
        if  reset_epoch:
            self.epoch_steps = 0
            self.epoch_value_accum = 0
        if reset_report:
            self.report_value_accum = 0
            self.report_steps = 0

    def update_value(self, value: float):
        self.report_steps += 1
        self.epoch_steps += 1
        self.epoch_value_accum += value
        self.report_value_accum += value

class CalculatedMetric(Metric,ABC):
    def __init__(self, name: str):
        super().__init__(name)

    def update(self,truth: torch.Tensor, pred: torch.Tensor):
        value = self.calculate(truth, pred)
        self.update_value(value)

    @abstractmethod
    def calculate(self, truth: torch.Tensor, prediction: torch.Tensor) -> float:
        pass
