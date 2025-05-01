import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CosAEEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, latent_channels=3*2*64):
        super().__init__()
        ch = base_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(ch, ch*2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch*2, ch*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(ch*2, ch*4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch*4, ch*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(ch*4, latent_channels, kernel_size=1, stride=1),
            # No activation here
        )

    def forward(self, x):
        return self.encoder(x)

class CosAE(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 num_components=64,  # important for big outputs
                 base_channels=64,
                 variant=None):
        super().__init__()

        if variant is not None:
            variant = variant.lower()
            if variant == 's':
                base_channels = 32
                num_components = 16
            elif variant == 'b':
                base_channels = 64
                num_components = 32
            elif variant == 'l':
                base_channels = 128
                num_components = 96

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_components = num_components

        latent_channels = out_channels * 2 * num_components

        self.encoder = CosAEEncoder(
            in_channels=self.in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels
        )

        self.freqs = nn.Parameter(torch.zeros(num_components, 2))
        nn.init.uniform_(self.freqs, -0.05, 0.05)  # pixel space small init

    def forward(self, x, out_shape=None):
        latent = self.encoder(x)
        B, _, H_lat, W_lat = latent.shape

        if out_shape is None:
            H_out, W_out = x.shape[-2:]
        else:
            H_out, W_out = out_shape

        K = self.num_components
        C_out = self.out_channels

        # Create pixel coordinate grid
        yy = torch.linspace(0, H_out - 1, H_out, device=x.device, dtype=latent.dtype)
        xx = torch.linspace(0, W_out - 1, W_out, device=x.device, dtype=latent.dtype)
        Y, X = torch.meshgrid(yy, xx, indexing='ij')

        angle_base = (self.freqs[:, 0].unsqueeze(-1).unsqueeze(-1) * X +
                      self.freqs[:, 1].unsqueeze(-1).unsqueeze(-1) * Y)

        latent = latent.view(B, C_out * 2 * K, H_lat, W_lat)
        latent = F.interpolate(latent, size=(H_out, W_out), mode='bilinear', align_corners=False)
        latent = latent.view(B, C_out, 2 * K, H_out, W_out)

        Amp = latent[:, :, :K, :, :]
        Phase = latent[:, :, K:, :, :]

        angle_base = angle_base.unsqueeze(0).unsqueeze(0)

        angle = angle_base + Phase
        cos_vals = torch.cos(angle)

        out = (cos_vals * Amp).sum(dim=2)

        return out
