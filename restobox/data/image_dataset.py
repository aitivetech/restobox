import abc
import os
from typing import List, Sized

import torch
import torchvision
import turbojpeg
from PIL import Image
from torch import dtype
from torch.utils.data import Dataset
import torchvision.transforms.v2 as TVT2

from restobox.images.image_utilities import find_image_files_by_extensions, load_image_file

class ImageDataset(Dataset[torch.Tensor],Sized,abc.ABC):
    pass


class ImageFolderDataset(ImageDataset):
    def __init__(self,
                 root_paths: str | List[str],
                 mode: str | None = "RGB",
                 dtype: torch.dtype = torch.float32,
                 scale : bool = True,
                 remove_invalid: bool = True,
                 extensions: List[str] | None = None) -> None:
        self.root_paths = root_paths
        self.mode = mode
        self.dtype = dtype
        self.jpeg = turbojpeg.TurboJPEG()
        self.remove_invalid = remove_invalid

        image_paths : list[str] = []
        root_paths = root_paths if isinstance(root_paths, list) else [root_paths]

        for root_path in root_paths:
            image_paths += find_image_files_by_extensions(root_path, extensions)

        self.image_paths = image_paths
        self.transform = TVT2.Compose([
            TVT2.ToImage(),
            TVT2.ToDtype(self.dtype,scale=scale)
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.__get_item_pil(index)

    def __get_item_pil(self, index: int) -> torch.Tensor:
        image = load_image_file(self.image_paths[index], self.mode)

        if image is None:

            if self.remove_invalid:
                if os.path.exists(self.image_paths[index]):
                    os.remove(self.image_paths[index])

            return self.__getitem__(index + 1 % len(self.image_paths))

        return self.transform(image)