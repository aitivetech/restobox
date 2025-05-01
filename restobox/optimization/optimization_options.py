from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizationOptions:
    base_lr: float = 2e-4 / 3
    betas: tuple[float,float] = (0.95, 0.98)
    weight_decay: float = 1e-2
    clip_grad_norm: float|None = 1.0