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

            nn.Conv2d(ch * 4, latent_channels, kernel_size=1)
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

        self.freqs = nn.Parameter(torch.empty(num_components, 2))
        nn.init.uniform_(self.freqs, 0.0, math.pi)

        self.post_cosine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # Optional projection if input and output channels differ
        self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, out_shape=None):
        B, _, H_in, W_in = x.shape
        latent = self.encoder(x)

        if out_shape is None:
            H_out, W_out = H_in, W_in
        else:
            H_out, W_out = out_shape

        C = self.out_channels
        K = self.num_components

        latent = latent.reshape(B, C, 2 * K, latent.shape[2], latent.shape[3])
        Amp = latent[:, :, :K, :, :]
        Phase = latent[:, :, K:, :, :]

        Amp = F.interpolate(Amp.reshape(B, C * K, *Amp.shape[-2:]),
                            size=(H_out, W_out), mode='bilinear', align_corners=False)
        Phase = F.interpolate(Phase.reshape(B, C * K, *Phase.shape[-2:]),
                              size=(H_out, W_out), mode='bilinear', align_corners=False)
        Amp = Amp.reshape(B, C, K, H_out, W_out)
        Phase = Phase.reshape(B, C, K, H_out, W_out)

        yy = torch.arange(H_out, device=x.device, dtype=latent.dtype)
        xx = torch.arange(W_out, device=x.device, dtype=latent.dtype)
        Y, X = torch.meshgrid(yy, xx, indexing='ij')

        freqs = self.freqs
        angle_base = freqs[:, 0].unsqueeze(-1).unsqueeze(-1) * X + \
                     freqs[:, 1].unsqueeze(-1).unsqueeze(-1) * Y
        angle_base = 2 * math.pi * angle_base
        angle_base = angle_base.unsqueeze(0).unsqueeze(0)

        angle = angle_base + Phase
        cos_vals = torch.cos(angle)
        out = (cos_vals * Amp).sum(dim=2)
        out = self.post_cosine(out)

        # 🔧 Residual connection
        skip = F.interpolate(x, size=(H_out, W_out), mode='bilinear', align_corners=False)
        skip = self.skip_proj(skip)
        out = out + skip

        return out


class CosAESR(nn.Module):

    def __init__(self,inner: CosAE,output_size: tuple[int,int]):
        super().__init__()
        self._inner = inner
        self._output_size = output_size

    def forward(self,x):
        return self._inner.forward(x,out_shape=self._output_size)

