import torch

from restobox.core.tensors import TensorLayout, TensorAxis, TensorAxisType, create_image_batch_layout
from restobox.models.model import Model
from restobox.tasks.sr.sr_image_task_options import ScaleFactor


def create_color_model(root: torch.nn.Module,
                    use_lab: bool,
                    input_size: tuple[int,int],
                    device: torch.device) -> Model:

    output_channels = 2 if use_lab else 3
    input_tensor = create_image_batch_layout(num_channels=1)
    output_tensor = create_image_batch_layout(num_channels=output_channels)

    return Model(root,device,[('input',input_tensor)], [('output',output_tensor)],False)