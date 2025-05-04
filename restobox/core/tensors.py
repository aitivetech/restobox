from dataclasses import dataclass
from enum import Enum, unique
from typing import Sequence

import numpy as np
import torch


@unique
class TensorAxisType(Enum):
    Time = 1
    Batch = 2
    Channel = 3
    Width = 4
    Height = 5
    Depth = 6,
    Class = 7 # OneHot encoded class probabilities

@dataclass
class TensorAxis:
    index: int
    type: TensorAxisType
    size: int | None = None
    min_size: int | None = 1
    max_size: int | None = None
    dynamic: bool = False

    @property
    def export_name(self) -> str:
        return self.type.name.lower()

@dataclass
class TensorLayout:
    axes: list[TensorAxis]
    dtype: torch.dtype | None = torch.float32

    @property
    def shape(self) -> Sequence[int]:
        return [axis.size for axis in self.axes]

    def create_random(self,device : torch.device,dtype: torch.dtype | None = None) -> torch.Tensor:
        actual_dtype = dtype if dtype is not None else self.dtype

        if actual_dtype is None:
            actual_dtype = torch.float32

        return torch.rand(self.shape, device=device, dtype=actual_dtype)

def create_image_batch_layout(num_channels:int,
                              size:tuple[int,int] = (1080,1920),
                              size_dynamic:bool = True,
                              max_size: tuple[int,int] = (1920*4,1080*4),
                              max_batch_size: int = 1024,
                              batch_dynamic: bool = True) -> TensorLayout:
    result = TensorLayout([
        TensorAxis(0,TensorAxisType.Batch,1,1,max_batch_size,batch_dynamic),
        TensorAxis(1,TensorAxisType.Channel,num_channels,num_channels,num_channels,False),
        TensorAxis(2,TensorAxisType.Height,size[0],32,max_size[0],size_dynamic),
        TensorAxis(3,TensorAxisType.Width,size[1],32,max_size[1],size_dynamic),
    ])

    return result

def tensor_to_uint8_numpy(t: torch.Tensor) -> np.ndarray:
    """
    Converts a float32 torch tensor (B, C, H, W) with values in [0,1] (but possibly with outliers)
    into a uint8 numpy array in shape (B, H, W, C).
    """
    t = t.detach()
    # Clamp values to [0, 1] to handle outliers
    t = torch.clamp(t, 0.0, 1.0)

    # Convert to [0, 255] and round
    t = (t * 255.0).round()

    # Change from (B, C, H, W) to (B, H, W, C) for standard image format
    #t = t.permute(0, 2, 3, 1)

    # Convert to uint8 numpy
    return t.byte().cpu().numpy()

