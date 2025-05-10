import abc
import os
from typing import List, Sized

import cv2
import numpy as np
import torch
import torchvision
import turbojpeg
from PIL import Image
from torch import dtype
from torch.utils.data import Dataset
import torchvision.transforms.v2 as TVT2

from restobox.images.image_utilities import find_image_files_by_extensions


class ImageDataset(Dataset[np.ndarray],Sized,abc.ABC):
    pass


class ImageFolderDataset(ImageDataset):
    def __init__(self,
                 root_paths: str | List[str],
                 remove_invalid: bool = True,
                 extensions: List[str] | None = None,
                 limit: int | None = None) -> None:
        self.root_paths = root_paths
        self.remove_invalid = remove_invalid

        image_paths : list[str] = []
        root_paths = root_paths if isinstance(root_paths, list) else [root_paths]

        for root_path in root_paths:
            image_paths += find_image_files_by_extensions(root_path, extensions)

        self.image_paths = image_paths

        if limit is not None:
            self.image_paths = self.image_paths[:limit]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> np.ndarray:
        return self._get_item_cv2(index)

    def _get_item_cv2(self,index: int) -> np.ndarray:

        image_path = self.image_paths[index]
        try:
            image = cv2.imread(self.image_paths[index]).astype(np.float32) / 255.
            return image
        except Exception as e:
            # If any error happens (corrupted image, etc.), skip and try another
            print(f"Warning: Failed to load image {image_path}. Skipping. Error: {e}")
            # Pick another random image instead
            if self.remove_invalid:
                if os.path.exists(self.image_paths[index]):
                    os.remove(self.image_paths[index])

            return self.__getitem__(index + 1 % len(self.image_paths))
