import torch
from lpips import LPIPS
from torch import nn


class LPipsAlex(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lpips = LPIPS(net='alex',spatial=True)

    def forward(self, prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        return self.lpips(prediction, truth).mean()