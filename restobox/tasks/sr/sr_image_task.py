import torch
import torchvision.transforms.v2.functional as Tvtf
from torchvision.transforms import InterpolationMode

from restobox.data.image_dataset import ImageDataset
from restobox.images.image_utilities import crop_and_resize
from restobox.models.model import Model
from restobox.tasks.image_task import ImageTask, Batch
from restobox.tasks.sr.sr_image_task_options import SrImageTaskOptions, mul_size


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

            high_res = crop_and_resize(item,output_crop_size,output_resize_size)
            low_res = Tvtf.resize(high_res,list(scale_factor.input_resize_size),interpolation=InterpolationMode.BICUBIC)

            high_res_items.append(high_res)
            low_res_items.append(low_res)

        return torch.stack(low_res_items),torch.stack(high_res_items)

    def create_baseline(self,
                        input_batch: torch.Tensor,
                        truth_batch: torch.Tensor) -> torch.Tensor | None:
        scale_factor = self.options.find_scale(self.global_step)
        return Tvtf.resize(input_batch,list(scale_factor.output_size),interpolation=InterpolationMode.BICUBIC)
