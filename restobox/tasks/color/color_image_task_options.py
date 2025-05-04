from dataclasses import dataclass

from restobox.tasks.image_task_options import ImageTaskOptions


@dataclass(frozen=True)
class ColorImageTaskOptions(ImageTaskOptions):
    use_lab: bool = True
    crop_size: tuple[int, int] = (320, 320)
    resize_size: tuple[int, int] = (256, 256)