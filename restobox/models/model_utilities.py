import torch

from restobox.models.model import Model
from restobox.training.training_options import TrainingOptions

_cpu_device = torch.device('cpu')

def prepare_model(model: Model,
                  training_options: TrainingOptions,
                  target_device: torch.device,
                  train: bool) -> tuple[Model,Model]:
    base_model = model.clone(device=_cpu_device)
    runtime_model = base_model.clone(device=target_device)

    runtime_model = runtime_model.compile() if training_options.compile_model else runtime_model

    if train:
        base_model.train()
        runtime_model.train()
    else:
        base_model.eval()
        runtime_model.eval()

    return base_model,runtime_model