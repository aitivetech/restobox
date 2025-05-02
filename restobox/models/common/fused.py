import torch
from torch import nn

from restobox.models.common.edsr import edsr
from restobox.models.common.rcan import rcan
from restobox.models.common.rdn import rdn_b


class SRFusion(nn.Module):
    def __init__(self, scale=4, pretrained=True):
        super().__init__()

        # Load pretrained SR models
        self.model1 = rcan(scale=scale, pretrained=pretrained)
        self.model2 = edsr(scale=scale, pretrained=pretrained)
        self.model3 = rdn_b(scale=scale, pretrained=pretrained)

        # Optional: freeze base models
        for model in [self.model1, self.model2, self.model3]:
            for param in model.parameters():
                param.requires_grad = False

        # Fusion head (learnable)
        self.fusion = nn.Sequential(
            nn.Conv2d(3 * 3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        out3 = self.model3(x)

        # Concatenate outputs along channel dimension
        fusion_input = torch.cat([out1, out2, out3], dim=1)  # (B, 9, H, W)
        fused = self.fusion(fusion_input)
        return fused