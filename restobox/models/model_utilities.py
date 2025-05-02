import torch
from torch.hub import load_state_dict_from_url

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

def load_pretrained_state(model, map_location,url: str | None = None):
    if model.url is None and url is None:
        raise KeyError("No URL available for this model")

    final_url = url if url is not None else model.url

    state_dict = load_state_dict_from_url(
        final_url, map_location=map_location, progress=True
    )
    model.load_state_dict(state_dict)