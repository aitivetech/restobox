from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizationOptions:
    base_lr: float = 2e-4 #2e-4 / 3
    betas: tuple[float,float] = (0.9, 0.999)
    weight_decay: float = 0.05
    clip_grad_norm: float|None = 1.0