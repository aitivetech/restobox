from enum import Enum

import torch


class ColorChannel(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3
    LUMINANCE = 4
    ALPHA = 5
    BETA = 6


class ColorPrecision(Enum):
    SDR = 1
    HDR = 2


class ColorChannelLayout(Enum):
    RGB = ([ColorChannel.RED, ColorChannel.GREEN, ColorChannel.BLUE])
    LAB = ([ColorChannel.RED, ColorChannel.GREEN, ColorChannel.BLUE])
    L = ([ColorChannel.LUMINANCE, ColorChannel.ALPHA, ColorChannel.BETA])
    AB = ([ColorChannel.ALPHA, ColorChannel.BETA])
    def __init__(self, channels: list[ColorChannel]):
        self.channels = channels


class ColorTensorLayout(Enum):

    RgbSdrInt = (ColorSpace.Rgb,ColorPrecision.Sdr,0,255,True)
    RgbHdrInt = (ColorSpace.Rgb,ColorPrecision.Hdr,0,255*255,True)


    def __init__(self,
                 space: ColorSpace,
                 precision: ColorPrecision,
                 min_value: float,
                 max_value: float,
                 is_integer: bool = False):
        self.space = space
        self.precision = precision
        self.min_value = min_value
        self.max_value = max_value
        self.is_integer = is_integer
        self.is_float = not is_integer
