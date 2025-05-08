import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CosAEEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, latent_channels=3 * 2 * 64):
        super().__init__()
        ch = base_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(ch, ch * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch * 2, ch * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(ch * 2, ch * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch * 4, ch * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(ch * 4, latent_channels, kernel_size=1, stride=1)
            # No activation
        )

    def forward(self, x):
        return self.encoder(x)


class CosAE(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 num_components=64,
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
        self.latent_channels = out_channels * 2 * num_components

        self.encoder = CosAEEncoder(
            in_channels=self.in_channels,
            base_channels=base_channels,
            latent_channels=self.latent_channels
        )

        # Frequencies: shape (K, 2) for u and v components
        self.freqs = nn.Parameter(torch.empty(num_components, 2))
        nn.init.uniform_(self.freqs, 0.0, 1.0)  # normalized freq for better high-freq coverage

    def forward(self, x, out_shape=None):
        B, _, H_in, W_in = x.shape
        latent = self.encoder(x)

        if out_shape is None:
            H_out, W_out = H_in, W_in
        else:
            H_out, W_out = out_shape

        C = self.out_channels
        K = self.num_components

        # Reshape latent into amplitude and phase
        latent = latent.view(B, C, 2 * K, latent.shape[2], latent.shape[3])

        # Split before interpolation
        Amp = latent[:, :, :K, :, :]  # [B, C, K, H_lat, W_lat]
        Phase = latent[:, :, K:, :, :]  # [B, C, K, H_lat, W_lat]

        # Interpolate each separately
        Amp = F.interpolate(Amp.reshape(B, C * K, *Amp.shape[-2:]), size=(H_out, W_out), mode='bilinear',
                            align_corners=False)
        Phase = F.interpolate(Phase.reshape(B, C * K, *Phase.shape[-2:]), size=(H_out, W_out), mode='bilinear',
                              align_corners=False)

        Amp = Amp.reshape(B, C, K, H_out, W_out)
        Phase = Phase.reshape(B, C, K, H_out, W_out)

        # Create normalized coordinate grid in range [0, 1]
        yy = torch.linspace(0, 1, H_out, device=x.device, dtype=latent.dtype)
        xx = torch.linspace(0, 1, W_out, device=x.device, dtype=latent.dtype)
        Y, X = torch.meshgrid(yy, xx, indexing='ij')  # [H, W]

        # Prepare the angle base from learnable frequencies
        freqs = self.freqs * 2 * math.pi  # scale to full 2π frequency domain
        angle_base = freqs[:, 0].unsqueeze(-1).unsqueeze(-1) * X + \
                     freqs[:, 1].unsqueeze(-1).unsqueeze(-1) * Y  # [K, H, W]

        angle_base = angle_base.unsqueeze(0).unsqueeze(0)  # [1, 1, K, H, W]
        angle = angle_base + Phase  # [B, C, K, H, W]
        cos_vals = torch.cos(angle)

        out = (cos_vals * Amp).sum(dim=2)  # sum over K

        return out


class CosAESR(nn.Module):

    def __init__(self,inner: CosAE,output_size: tuple[int,int]):
        super().__init__()
        self._inner = inner
        self._output_size = output_size

    def forward(self,x):
        return self._inner.forward(x,out_shape=self._output_size)

