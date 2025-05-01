import warnings
from enum import Enum, StrEnum

import torch
from PIL import Image


class PerformanceMode(StrEnum):
    SLOW = 'slow'
    MEDIUM = 'medium'
    FAST = 'fast'

def optimize_performance(performance_mode: PerformanceMode):
    precision = 'highest'

    if performance_mode == PerformanceMode.FAST:
        precision = 'medium'
    elif performance_mode == PerformanceMode.MEDIUM:
        precision = 'high'

    torch.set_float32_matmul_precision(precision=precision)

    if performance_mode != PerformanceMode.SLOW:
        torch.backends.cudnn.benchmark = True

def disable_warnings():
    warnings.filterwarnings("ignore", message="the float32 number .* will be truncated to .*")

    warnings.filterwarnings(
        action="ignore",
        category=FutureWarning,
        message=".* is deprecated in version 0.1*"
    )

    warnings.simplefilter('ignore', Image.DecompressionBombWarning)
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message="explain\\(f, \\*args, \\*\\*kwargs\\) is deprecated")
