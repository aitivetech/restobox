import abc
import json
import os
from typing import Any

import kornia.losses
import lpips
import timm.optim
import torch
from torch import GradScaler
from torch.nn import L1Loss, MSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from restobox.data.image_dataset import ImageDataset
from restobox.diagnostics.memory import get_memory_info
from restobox.diagnostics.profiler import Profiler
from restobox.export.export_onnx import export_onnx, convert_fp16, convert_int8_dynamic
from restobox.export.export_options import ExportOptions
from restobox.losses.charbonnier_loss import CharbonnierLoss
from restobox.losses.lpips_loss import LPipsAlex
from restobox.losses.perceptual_loss import CombinedPerceptualLoss,ChainedLoss, ChainedLossEntry
from restobox.metrics.external_metric import ExternalMetric
from restobox.metrics.metric import Metric, CalculatedMetric
from restobox.metrics.psnr_metric import PsnrMetric
from restobox.metrics.ssim_metric import SsimMetric
from restobox.models.model import Model
from restobox.models.model_utilities import prepare_model
from restobox.optimization.optimization_options import OptimizationOptions
from restobox.reporting.progress_report_writer import ProgressReportWriter
from restobox.reporting.report_writer import ChainedReportWriter
from restobox.reporting.visdom_report_writer import VisdomReportWriter
from restobox.tasks.task_options import TaskOptions
from restobox.tasks.task import Task
from restobox.training import training_options
from restobox.training.training_options import TrainingOptions
from restobox.training.training_utilities import optimize_performance

type Batch = tuple[torch.Tensor, torch.Tensor]


class ImageTask(Task):
    def __init__(self,
                 dataset: ImageDataset,
                 model: Model,
                 options: TaskOptions,
                 device: torch.device) -> None:
        super().__init__(dataset, model, options, device)

    def create_loss(self) -> torch.nn.Module:
        return ChainedLoss([
            ChainedLossEntry(L1Loss(), weight=1),
            ChainedLossEntry(LPipsAlex(), weight=0.05),
        ])

    def create_metrics(self) -> list[Metric]:
        return [PsnrMetric(), SsimMetric()]

    def evaluate(self, report_writer, input_batch, prediction_batch, truth_batch):

        result = self.create_results(input_batch, truth_batch, prediction_batch)
        baseline = self.create_baseline(input_batch, truth_batch)

        images = dict()

        if baseline is not None:
            # Stack in order [baseline[i], prediction[i], truth[i]] for each i
            interleaved = torch.stack([baseline, result, truth_batch], dim=1)  # shape: (B, 3, C, H, W)
            interleaved = interleaved.view(-1, *baseline.shape[1:])
            images["baseline,prediction,truth"] = interleaved

        report_writer.update_images(self.epoch, self.step_in_epoch, self.global_step, images)







