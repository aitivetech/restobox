from typing import List

import pillow_jxl
from PIL import Image

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
