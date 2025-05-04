import torch

from restobox.core.tensors import TensorLayout, TensorAxis, TensorAxisType, create_image_batch_layout
from restobox.models.model import Model
from restobox.tasks.sr.sr_image_task_options import ScaleFactor, mul_size


def create_sr_model(root: torch.nn.Module,
                    scale: ScaleFactor,
                    device: torch.device) -> Model:

    max_size = (1080,1920)
    input_tensor = create_image_batch_layout(num_channels=3,size=scale.input_size,max_size=max_size)
    output_tensor = create_image_batch_layout(num_channels=3,size=scale.output_size,max_size=mul_size(max_size,scale.factor))

    return Model(root,device,[('input',input_tensor)], [('output',output_tensor)],False)