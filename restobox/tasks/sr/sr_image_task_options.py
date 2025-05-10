from dataclasses import dataclass
from typing import List

from restobox.tasks.image_task_options import ImageTaskOptions
from restobox.tasks.task_options import TaskOptions

def mul_size(size: tuple[int,int], factor: float | int) -> tuple[int,int]:
    return int(size[0] * factor), int(size[1] * factor)

@dataclass(frozen=True)
class ScaleFactor:
    factor: int
    start_step: int
    input_resize_size: tuple[int,int]
    input_crop_size: tuple[int,int]
    use_bsrgan: bool = True

    @property
    def output_size(self) -> tuple[int,int]:
        return mul_size(self.input_resize_size, self.factor)

    @property
    def input_size(self) -> tuple[int,int]:
        return self.input_resize_size


    @staticmethod
    def simple(factor: int,input_size: tuple[int,int] | int,start_step: int) -> 'ScaleFactor':
        input_width = input_size[1] if isinstance(input_size, tuple) else input_size
        input_height = input_size[0] if isinstance(input_size, tuple) else input_size

        input_crop_size = mul_size((input_height, input_width), 1.25)
        input_resize_size = (input_height, input_width)

        return ScaleFactor(factor, start_step, input_resize_size,input_crop_size)



@dataclass(frozen=True)
class SrImageTaskOptions(ImageTaskOptions):
    scales: List[ScaleFactor]
    augment: bool = True


    def find_scale(self,step: int) -> ScaleFactor:
        for scaler in reversed(self.scales):
            if scaler.start_step <= step:
                return scaler
        raise ValueError(f"Scale factor not found for step {step}")
