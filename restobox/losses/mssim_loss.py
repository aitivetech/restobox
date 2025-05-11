import torch
from pytorch_msssim import ms_ssim
from torch import nn


class MSSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,prediction:torch.Tensor,truth:torch.Tensor) -> torch.Tensor:
        return 1 - ms_ssim(prediction, truth, data_range=255, size_average=True)