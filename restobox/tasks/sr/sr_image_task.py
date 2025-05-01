import torch
from torchvision.transforms import InterpolationMode

from restobox.data.image_dataset import ImageDataset
from restobox.export.export_options import ExportOptions
from restobox.models.model import Model
from restobox.optimization.optimization_options import OptimizationOptions
from restobox.tasks.image_task import ImageTask, Batch
from restobox.tasks.sr.sr_image_task_options import SrImageTaskOptions, mul_size
from restobox.training.training_options import TrainingOptions

import torchvision.transforms.v2.functional as Tvtf

class SrImageTask(ImageTask):

    def __init__(self,
                 dataset: ImageDataset,
                 model: Model,
                 options: SrImageTaskOptions,
                 device: torch.device):
        super().__init__(dataset, model, options, device)
        self.options = options

    def create_batch(self, items) -> Batch:
        scale_factor = self.options.find_scale(self.global_step)

        high_res_items: list[torch.Tensor] = []
        low_res_items: list[torch.Tensor] = []

        output_crop_size = mul_size(scale_factor.input_crop_size,scale_factor.factor)
        output_resize_size = mul_size(scale_factor.input_resize_size,scale_factor.factor)

        for item in items:
            high_res = Tvtf.center_crop(item,list(output_crop_size))
            high_res = Tvtf.resize(high_res,list(output_resize_size),interpolation=InterpolationMode.BICUBIC)

            low_res = Tvtf.resize(high_res,list(scale_factor.input_resize_size),interpolation=InterpolationMode.BICUBIC)

            high_res_items.append(high_res)
            low_res_items.append(low_res)

        return torch.stack(low_res_items),torch.stack(high_res_items)

    def create_baseline(self,
                        input_batch: torch.Tensor,
                        truth_batch: torch.Tensor) -> torch.Tensor | None:
        scale_factor = self.options.find_scale(self.global_step)
        return Tvtf.resize(input_batch,list(scale_factor.output_size),interpolation=InterpolationMode.BICUBIC)
