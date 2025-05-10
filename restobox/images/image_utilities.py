import math
from typing import List

import cv2
import numpy as np
import pillow_jxl
import torch
from PIL import Image
import torchvision.transforms.v2.functional as Tvtf
from torchvision.transforms import InterpolationMode

from restobox.data.file_utilities import find_files_by_extensions

IMAGE_FILE_EXTENSIONS = ['.jpg', 'jpeg', 'png', 'webp','.jxl']


def find_image_files_by_extensions(
        directory: str,
        extensions=None,
        recursive: bool = True
) -> List[str]:
    if extensions is None:
        extensions = IMAGE_FILE_EXTENSIONS
    return find_files_by_extensions(directory, extensions, recursive)


def load_image_file(image_path: str, mode: str | None = "RGB") -> Image.Image | None:
    try:
        with Image.open(image_path) as img:
            img = img.convert(mode)
            return img
    except Exception as e:
        # If any error happens (corrupted image, etc.), skip and try another
        print(f"Warning: Failed to load image {image_path}. Skipping. Error: {e}")
        # Pick another random image instead
        return None

def crop_and_resize(image: torch.Tensor,crop_size: tuple[int,int],resize_size: tuple[int,int]) -> torch.Tensor:
    if image.shape[1] != resize_size[0] or image.shape[2] != resize_size[1]:
        image = Tvtf.center_crop(image, list(crop_size))
        image = Tvtf.resize(image, list(resize_size), interpolation=InterpolationMode.BICUBIC)

    return image

def np_uint_to_tensor4(img: np.ndarray) -> torch.Tensor:
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.).unsqueeze(0)



