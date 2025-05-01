import torch
from torch.nn import functional as F

from restobox.metrics.metric import CalculatedMetric

def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = F.mse_loss(pred, target, reduction='mean')
    psnr_value = 10 * torch.log10(max_val ** 2 / mse)
    return psnr_value

class PsnrMetric(CalculatedMetric):
    def __init__(self, name: str= "PSNR"):
        super().__init__(name)

    def calculate(self, truth: torch.Tensor, prediction: torch.Tensor) -> float:
        return psnr(prediction, truth).mean().item()
