import copy
from typing import Any

import torch
from torch import nn
from torch._dynamo.backends.debugging import ExplainOutput

from restobox.core.tensors import TensorLayout

type ModelIoSpec = list[tuple[str,TensorLayout]]

class Model:
    def __init__(self,
                 root: nn.Module,
                 device: torch.device,
                 inputs: ModelIoSpec,
                 outputs: ModelIoSpec,
                 is_compiled: bool = False) -> None:
        self.root = root
        self.device = device
        self.inputs = inputs
        self.outputs = outputs
        self.is_compiled = is_compiled

    def train(self):
        self.root.train()

    def eval(self):
        self.root.eval()

    @property
    def state_dict(self) -> dict[str, Any]:
        return self.root._orig_mod.state_dict() if self.is_compiled else self.root.state_dict()

    def copy_state_to(self, model: 'Model') -> None:
        model.root.load_state_dict(self.state_dict)

    def load_state(self,state: Any)->None:
        self.root.load_state_dict(state)

    def clone(self,
              device: torch.device | None = None,
              dtype: torch.dtype | None = None) -> "Model":
        self._disallow_compiled('clone')
        root = copy.deepcopy(self.root)

        if device is not None:
            root.to(device=device)

        if dtype is not None:
            root.to(dtype=dtype)

        return Model(root, device if device is not None else self.device, self.inputs,self.outputs,self.is_compiled)

    def compile(self,**kwargs) -> 'Model':
        compiled_root = torch.compile(model=self.root,
                                      fullgraph=True,
                                      backend='inductor',
                                      mode='max-autotune',**kwargs)
        return Model(compiled_root,self.device,self.inputs,self.outputs,True)

    def get_compilation_diagnostics(self,*args) -> ExplainOutput:
        if not self.is_compiled:
            raise ValueError("Only works on compiled models")

        explanation = torch._dynamo.explain(self.root, *args)
        return explanation

    @torch.no_grad()
    def update_ema(self, ema_model: 'Model', decay=0.9999) -> None:

        model_state = self.state_dict
        ema_state = ema_model.state_dict

        for key in ema_state.keys():
            if key in model_state:
                model_param = model_state[key]
                ema_param = ema_state[key]
                if model_param.dtype.is_floating_point:
                    # Normal EMA update for float parameters
                    ema_param.mul_(decay).add_(model_param, alpha=1 - decay)
                else:
                    # For non-float params (int, long, etc.), just copy
                    ema_param.copy_(model_param)


    def _disallow_compiled(self,task : str):
        if self.is_compiled:
            raise RuntimeError(f'Unable to apply {task} to compiled model')