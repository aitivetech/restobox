import torch

from restobox.core.tensors import TensorLayout, TensorAxis, TensorAxisType
from restobox.models.model import Model
from restobox.tasks.sr.sr_image_task_options import ScaleFactor


def create_sr_model(root: torch.nn.Module,
                    scale: ScaleFactor,
                    device: torch.device) -> Model:

    max_size = 1920 * 2

    input_tensor = TensorLayout([
        TensorAxis(0,TensorAxisType.Batch,1,1,1024,True),
        TensorAxis(1,TensorAxisType.Channel,3,3,3,False),
        TensorAxis(2,TensorAxisType.Height,scale.input_size[0],32,max_size,True),
        TensorAxis(3,TensorAxisType.Width,scale.input_size[1],32,max_size,True),
    ])

    output_tensor = TensorLayout([
        TensorAxis(0, TensorAxisType.Batch, 1, 1, 1024, True),
        TensorAxis(1, TensorAxisType.Channel, 3, 3, 3, False),
        TensorAxis(2, TensorAxisType.Height, scale.output_size[0], 32, max_size * scale.factor, True),
        TensorAxis(3, TensorAxisType.Width, scale.output_size[1], 32, max_size * scale.factor, True),
    ])

    return Model(root,device,[('input',input_tensor)], [('output',output_tensor)],False)