import cv2
import torch
import torchvision.transforms.v2.functional as Tvtf
from torchvision.transforms import InterpolationMode

from restobox.data.image_dataset import ImageDataset
from restobox.images.degradation_utilities import random_crop_np, random_rotate, \
    random_horizontally_flip, random_vertically_flip, degradation_process_plus, image_to_tensor
from restobox.models.model import Model
from restobox.tasks.image_task import ImageTask, Batch
from restobox.tasks.sr.sr_image_task_options import SrImageTaskOptions, mul_size, ScaleFactor




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

        for item in items:

            output_cropped = random_crop_np(item,scale_factor.output_size[0])

            if self.options.augment:
                output_cropped = random_rotate(output_cropped, [90, 180, 270])
                output_cropped = random_horizontally_flip(output_cropped, 0.5)
                output_cropped = random_vertically_flip(output_cropped, 0.5)

            output_cropped = cv2.cvtColor(output_cropped, cv2.COLOR_BGR2RGB)

            if scale_factor.use_bsrgan:
                lr_cropped =  degradation_process_plus(output_cropped, int(scale_factor.factor),gray_prob=0.0)
            else:
                lr_cropped = cv2.resize(output_cropped,scale_factor.input_size,interpolation=cv2.INTER_CUBIC)

            lr = image_to_tensor(lr_cropped,False,False)
            hr = image_to_tensor(output_cropped,False,False)

            high_res_items.append(hr)
            low_res_items.append(lr)

        return torch.stack(low_res_items),torch.stack(high_res_items)

    def create_baseline(self,
                        input_batch: torch.Tensor,
                        truth_batch: torch.Tensor) -> torch.Tensor | None:
        scale_factor = self.options.find_scale(self.global_step)
        return Tvtf.resize(input_batch,list(scale_factor.output_size),interpolation=InterpolationMode.BICUBIC)

