import torch
from torch import nn
from transformers import Swin2SRConfig, Swin2SRForImageSuperResolution

from restobox.models.common.swinirv2 import Swin2SR

class Swin2SRUnpacker(nn.Module):
    def __init__(self,swin2sr:Swin2SR):
        super().__init__()
        self.swin2sr = swin2sr

    def forward(self,input_batch: torch.Tensor) -> torch.Tensor:
        return self.swin2sr(input_batch)[0]

def create_swin2sr(input_size: tuple[int,int],scale_factor: int) -> nn.Module:
    configuration = Swin2SRConfig(image_size=input_size,upscale=scale_factor,return_dict=False)
    model = Swin2SRForImageSuperResolution(configuration)
    return Swin2SRUnpacker(model.swin2sr)