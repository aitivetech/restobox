from typing import Any

import torch

from restobox.core.tensors import TensorLayout


def create_dynamic_shape(layout: TensorLayout, use_dims: bool) -> dict[int, Any]:
    result: dict[int, str | Any] = {}

    for axis in layout.axes:
        if axis.dynamic:
            result[axis.index] = axis.export_name if not use_dims else torch.export.Dim(axis.export_name,
                                                                                        min=axis.min_size,
                                                                                        max=axis.max_size)
    return result
