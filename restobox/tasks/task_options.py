from abc import ABC
from dataclasses import dataclass

from restobox.export.export_options import ExportOptions
from restobox.optimization.optimization_options import OptimizationOptions
from restobox.training.training_options import TrainingOptions


@dataclass(frozen=True)
class TaskOptions(ABC):
    training: TrainingOptions
    optimization: OptimizationOptions
    export: ExportOptions