import torch
import torch.nn as nn
import torch.nn.functional as F
from lpips import LPIPS
from torchvision.models import vgg16, VGG16_Weights


class PerceptualLoss(nn.Module):
    def __init__(self, layer_index:int=12):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:layer_index].eval()
        for param in vgg.parameters():
            param.requires_grad = False

        self.vgg = vgg

    def forward(self, prediction : torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        # Assumes input is normalized to [0, 1]
        return F.l1_loss(self.vgg(prediction), self.vgg(truth))

class CombinedPerceptualLoss(nn.Module):
    def __init__(self,
                 base_loss:nn.Module = nn.L1Loss(),
                 perceptual_weight:float=0.01,
                 perceptual_enabled = True,
                 layer_index=12):
        super().__init__()

        self.perceptual_weight = perceptual_weight
        self.perceptual_enabled = perceptual_enabled
        self.perceptual = PerceptualLoss(layer_index=layer_index)
        self.base = base_loss

    def forward(self, prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        loss_base = self.base(prediction, truth)

        if self.perceptual_enabled:
            loss_perceptual = self.perceptual(prediction, truth)

            final_loss = loss_base + self.perceptual_weight * loss_perceptual
            return final_loss

        return loss_base

    def set_perceptual(self, perceptual_enabled: True, perceptual_weight:float | None) -> None:
        self.perceptual_enabled = perceptual_enabled
        self.perceptual_weight = self.perceptual_weight if perceptual_weight is None else perceptual_weight

class LPipsAlex(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lpips = LPIPS(net='alex')

    def forward(self, prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        return self.lpips(prediction, truth).mean()