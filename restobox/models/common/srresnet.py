import math

from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class SRResNet(nn.Module):
    def __init__(self, scale_factor:int=4, in_channels:int=3,out_channels:int=3, base_channels=64, num_blocks=8):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=9, padding=4),
            nn.ReLU()
        )
        self.body = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_blocks)])
        self.mid = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.tail = self._make_upsample(base_channels, scale_factor)
        self.output = nn.Conv2d(base_channels, out_channels, kernel_size=9, padding=4)

    def _make_upsample(self, channels, scale):
        layers = []
        while scale > 1:
            if scale % 2 != 0:
                raise ValueError("Scale factor must be a power of 2")
            layers.append(nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1))
            layers.append(nn.PixelShuffle(2))
            layers.append(nn.ReLU())
            scale //= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = self.mid(res) + x
        x = self.tail(x)
        return self.output(x)
