import torch
from kornia.color import rgb_to_lab, lab_to_rgb
from torch import nn
from torch.nn import L1Loss

from restobox.core.tensors import create_image_batch_layout, clone_image_batch_layout
from restobox.data.image_dataset import ImageDataset
from restobox.images.conversions import rgb_to_l_ab, l_ab_to_rgb
from restobox.images.image_utilities import crop_and_resize
from restobox.losses.charbonnier_loss import CharbonnierLoss
from restobox.models.model import Model
from restobox.tasks.color.color_image_task_options import ColorImageTaskOptions
from restobox.tasks.image_task import ImageTask
from restobox.tasks.task import Batch

class LabWrapper(nn.Module):
    def __init__(self, lab_model: nn.Module):
        super().__init__()
        self.model = lab_model

    def forward(self, rgb_input):
        lab_input = rgb_to_lab(rgb_input)

        l_input = lab_input[:,[0],...]
        l_normalized_input = l_input / 50.0 - 1.0

        ab_output = self.model(l_normalized_input)

        ab_denormalized_output = ab_output * 110.0
        lab_output = torch.cat([l_input, ab_denormalized_output], dim=1)

        rgb_output = lab_to_rgb(lab_output)
        return rgb_output.clamp(min=0, max=1.0)


class ColorImageTask(ImageTask):
    def __init__(self,
                 dataset: ImageDataset,
                 model: Model,
                 options: ColorImageTaskOptions,
                 device: torch.device) -> None:
        super().__init__(dataset, model, options, device)
        self.options = options

    def create_loss(self) -> torch.nn.Module:
        return CharbonnierLoss()

    def create_batch(self, items) -> Batch:
        inputs = []
        outputs = []

        for item in items:
            image = crop_and_resize(item,self.options.crop_size,self.options.resize_size)
            l,ab = rgb_to_l_ab(image)

            inputs.append(l)

            if self.options.use_lab:
                outputs.append(ab)
            else:
                outputs.append(image)

        return torch.stack(inputs),torch.stack(outputs)

    def create_results(self,
                       input_batch: torch.Tensor,
                       truth_batch: torch.Tensor,
                       predictions_batch: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        if self.options.use_lab:
            rgb_prediction = l_ab_to_rgb(input_batch, predictions_batch)
            rgb_truth = l_ab_to_rgb(input_batch,truth_batch)

            return rgb_prediction,rgb_truth

        return predictions_batch,truth_batch

    def get_export_model(self, model: Model) -> Model:

        root = model.root
        if self.options.use_lab:
            root = LabWrapper(root)

            new_input_layout = clone_image_batch_layout(model.inputs[0][1],3)
            return Model(root,model.device,[(model.inputs[0][0],new_input_layout)],model.outputs,model.is_compiled)

        return model

