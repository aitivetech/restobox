from dataclasses import dataclass

import torch

from restobox.training.training_utilities import PerformanceMode


@dataclass(frozen=True)
class TrainingOptions:
    batch_size: int
    epochs: int
    output_path: str

    limit_steps: int | None = None
    num_workers: int = 4
    prefetch_factor: int = 2
    performance: PerformanceMode = PerformanceMode.FAST
    compile_model: bool = True
    compile_explain: bool = True
    profile: bool = True

    profile_wait: int = 10
    profile_warmup: int = 10
    profile_active: int = 10
    profile_repeat: int = 1

    use_amp : bool = True
    amp_dtype: torch.dtype = torch.bfloat16

    checkpoint_every_n_steps: int = 1000
    checkpoint_best: bool = True
    evaluate_every_n_steps: int = 1000
