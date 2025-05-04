import torch
from torch.nn import L1Loss

from restobox.data.image_dataset import ImageDataset
from restobox.losses.lpips_loss import LPipsAlex
from restobox.losses.perceptual_loss import ChainedLoss, ChainedLossEntry
from restobox.metrics.metric import Metric
from restobox.metrics.psnr_metric import PsnrMetric
from restobox.metrics.ssim_metric import SsimMetric
from restobox.models.model import Model
from restobox.tasks.image_task_options import ImageTaskOptions
from restobox.tasks.task import Task

type Batch = tuple[torch.Tensor, torch.Tensor]


class ImageTask(Task):
    def __init__(self,
                 dataset: ImageDataset,
                 model: Model,
                 options: ImageTaskOptions,
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

        result_prediction,result_truth = self.create_results(input_batch, truth_batch, prediction_batch)
        baseline = self.create_baseline(input_batch, truth_batch)

        images = dict()

        if baseline is not None:
            interleaved = torch.stack([baseline, result_prediction, result_truth], dim=1)  # shape: (B, 3, C, H, W)
            interleaved = interleaved.view(-1, *baseline.shape[1:])
            images["baseline,prediction,truth"] = interleaved
        else:
            interleaved = torch.stack([result_prediction, result_truth], dim=1)  # shape: (B, 3, C, H, W)
            interleaved = interleaved.view(-1, *result_prediction.shape[1:])
            images["prediction,truth"] = interleaved

        report_writer.update_images(self.epoch, self.step_in_epoch, self.global_step, images)







