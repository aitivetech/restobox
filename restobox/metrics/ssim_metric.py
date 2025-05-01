import torch
from pytorch_msssim import ssim

from restobox.metrics.metric import CalculatedMetric


class SsimMetric(CalculatedMetric):
    def __init__(self,name:str='SSIM',data_range: float = 1.0):
        super().__init__(name)
        self.data_range = data_range

    def calculate(self, truth: torch.Tensor, prediction: torch.Tensor) -> float:
        return ssim(prediction,truth, data_range=self.data_range, size_average=True).item()
