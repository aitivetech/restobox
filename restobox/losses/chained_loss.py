import torch
from torch import nn as nn
from torch.nn import ModuleList


class ChainedLossEntry(nn.Module):
    def __init__(self,inner: nn.Module,weight:float,is_enabled: bool = True):
        super().__init__()
        self.inner = inner
        self.weight = weight
        self.is_enabled = is_enabled

    def forward(self,prediction: torch.Tensor,truth: torch.Tensor) -> torch.Tensor:
        return self.weight * self.inner(prediction,truth)


class ChainedLoss (nn.Module):
    def __init__(self,
                losses: list[ChainedLossEntry]):
        super().__init__()
        self.losses = ModuleList(losses)

    def set_loss_state(self,
                       index:int,
                       enabled: bool | None = None,
                       weight:float | None = None):
        loss = self.losses[index]

        if weight is not None:
            loss.weight = weight

        if enabled is not None:
            loss.is_enabled = enabled

    def forward(self,prediction: torch.Tensor,truth: torch.Tensor) -> torch.Tensor:
        result = torch.tensor(0,dtype=prediction.dtype,device=prediction.device)

        for loss in self.losses:
            if loss.is_enabled:
                result += loss(prediction,truth)

        return result
