import abc
import json
import os
from typing import Any

import torch
from torch import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from restobox.data.image_dataset import ImageDataset
from restobox.diagnostics.memory import get_memory_info
from restobox.diagnostics.profiler import Profiler
from restobox.export.export_onnx import export_onnx, convert_fp16, convert_int8_dynamic
from restobox.metrics.external_metric import ExternalMetric
from restobox.metrics.metric import Metric, CalculatedMetric
from restobox.models.model import Model
from restobox.models.model_utilities import prepare_model
from restobox.reporting.progress_report_writer import ProgressReportWriter
from restobox.reporting.report_writer import ChainedReportWriter
from restobox.reporting.visdom_report_writer import VisdomReportWriter
from restobox.tasks.task_options import TaskOptions
from restobox.training.training_utilities import optimize_performance

type Batch = tuple[torch.Tensor, torch.Tensor]


class Task(abc.ABC):
    def __init__(self,
                 dataset: ImageDataset,
                 model: Model,
                 options: TaskOptions,
                 device: torch.device) -> None:
        self.dataset = dataset

        if len(dataset) < 1:
            raise ValueError("Dataset is empty")

        self.options = options
        self.training_options = options.training
        self.optimization_options = options.optimization
        self.export_options = options.export

        self.base_model, self.train_model = prepare_model(model, self.training_options, device, True)
        self.ema_base_model, self.ema_train_model = prepare_model(model, self.training_options, device, False)

        self.device = device

        self.steps_per_epoch = len(dataset) // self.training_options.batch_size
        self.total_steps = len(dataset) * self.training_options.epochs / self.training_options.batch_size
        self.num_items = len(dataset)

        self.run_id = "unknown_run"
        self.step_in_epoch = 0
        self.global_step = 0
        self.epoch = 1

        self.optimizer = self.create_optimizer(self.train_model)
        self.scheduler = self.create_scheduler(self.train_model, self.optimizer)
        self.criterion = self.create_loss().to(device)

        self.loss_metric = ExternalMetric("Loss")
        self.free_memory_metric = ExternalMetric("FreeMemory", ".0f")
        self.metrics = self.create_metrics()
        self.metrics.append(self.loss_metric)
        self.metrics.append(self.free_memory_metric)

        self.best_loss_value = 1000000

    def train(self, run_id: str):

        optimize_performance(self.training_options.performance)
        self.run_id = run_id

        os.makedirs(self._get_output_path(), exist_ok=True)

        print(f"{run_id}: Training for {self.total_steps} steps on {self.num_items} items")
        print(f"Results at: {self._get_output_path()}")

        has_been_evaluated_once = False

        effective_dtype = self.training_options.amp_dtype if self.training_options.use_amp else torch.float32

        dataloader = self.create_dataloader()
        scaler = GradScaler(device=self.device.type, enabled=self.training_options.use_amp)
        visdom_writer = VisdomReportWriter(env=run_id)

        with Profiler(enabled=self.training_options.profile, wait=self.training_options.profile_wait,
                      active=self.training_options.profile_active, repeat=self.training_options.profile_repeat,
                      output_json=self._get_output_path('profile_trace.json')) as profiler:

            for i in range(1, self.training_options.epochs + 1):

                self.epoch = i
                self.step_in_epoch = 0
                self._reset_metrics(True, True)

                progress_bar = tqdm(enumerate(dataloader), total=self.steps_per_epoch)
                report_writer = ChainedReportWriter([
                    visdom_writer,
                    ProgressReportWriter(progress_bar)
                ])

                for batch_idx, batch in progress_bar:
                    self.optimizer.zero_grad()

                    self.step_in_epoch += 1
                    self.global_step += 1

                    with torch.amp.autocast(
                            device_type=self.device.type,
                            dtype=self.training_options.amp_dtype,
                            enabled=self.training_options.use_amp):

                        input_batch = batch[0].to(device=self.device, dtype=effective_dtype, non_blocking=True)
                        truth_batch = batch[1].to(device=self.device, dtype=effective_dtype, non_blocking=True)

                        prediction_batch = self.train_model.root(input_batch)

                        if isinstance(prediction_batch,tuple):
                            prediction_batch = prediction_batch[0]

                        loss = self.criterion(prediction_batch, truth_batch)
                        loss_value = loss.item()

                        # Gradscale
                        scaler.scale(loss).backward()

                        if self.optimization_options.clip_grad_norm:
                            scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.train_model.root.parameters(),
                                                           self.optimization_options.clip_grad_norm)

                        scaler.step(self.optimizer)
                        scaler.update()

                        self.train_model.update_ema(self.ema_train_model)

                        self.scheduler.step()

                        profiler.step()

                        self._update_metrics(report_writer, loss_value, prediction_batch, truth_batch)

                        if self.global_step % self.training_options.evaluate_every_n_steps == 0:
                            self.evaluate(report_writer, input_batch, prediction_batch, truth_batch)

                            if not has_been_evaluated_once:
                                self._explain_compilation(input_batch)
                                has_been_evaluated_once = True

                        if self.global_step % self.training_options.checkpoint_every_n_steps == 0:
                            self._update_checkpoints(loss_value)

                        if self.training_options.limit_steps is not None:
                            if self.step_in_epoch > self.training_options.limit_steps:
                                continue

    def create_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.training_options.batch_size,
            num_workers=self.training_options.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            collate_fn=self.create_batch,
            prefetch_factor=self.training_options.prefetch_factor,
        )

    @abc.abstractmethod
    def create_batch(self, items) -> Batch:
        pass

    @abc.abstractmethod
    def create_loss(self) -> torch.nn.Module:
        pass

    @abc.abstractmethod
    def create_metrics(self) -> list[Metric]:
        pass

    def create_results(self,
                       input_batch: torch.Tensor,
                       truth_batch: torch.Tensor,
                       predictions_batch: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        return predictions_batch,truth_batch

    def create_baseline(self,
                        input_batch: torch.Tensor,
                        truth_batch: torch.Tensor) -> torch.Tensor | None:
        return None

    def create_optimizer(self, model: Model) -> torch.optim.Optimizer:
        #return timm.optim.Lion(model.root.parameters(), lr=self.optimization_options.base_lr,
         #                      weight_decay=self.optimization_options.weight_decay,
          #                     betas=self.optimization_options.betas)

        return AdamW(model.root.parameters(),
                     fused=True,
                     lr=self.optimization_options.base_lr,
                     weight_decay=self.optimization_options.weight_decay,
                     betas=self.optimization_options.betas)

    def create_scheduler(self, model: Model, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.optimization_options.base_lr,  # target peak LR
            total_steps=int(self.total_steps),  # total number of training steps
            pct_start=0.1,  # % of total steps used for warmup
            anneal_strategy='cos',  # cosine annealing after warmup
            div_factor=100.0,  # initial LR = max_lr / div_factor
            final_div_factor=1e4  # final LR = max_lr / final_div_factor
        )

        return scheduler

    def evaluate(self, report_writer, input_batch, prediction_batch, truth_batch):
        pass

    def get_export_model(self, model: Model) -> Model:
        return model

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path)

        self.base_model.load_state(checkpoint['model_state_dict'])
        self.ema_base_model.load_state(checkpoint['ema_model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.global_step = checkpoint['global_step']
        self.step_in_epoch = checkpoint['step_in_epoch']
        self.epoch = checkpoint['epoch']

    def _update_checkpoints(self, loss_value: float):
        directory = f"step_{self.global_step}_epoch_{self.epoch}"
        self._save_checkpoint(loss_value, directory)

        if loss_value < self.best_loss_value:
            self._save_checkpoint(loss_value, "best")
            self.best_loss_value = loss_value

    def _save_checkpoint(self, loss_value: float, directory: str):
        self.train_model.copy_state_to(self.base_model)
        self.ema_train_model.copy_state_to(self.ema_base_model)

        manifest_data: dict[str, Any] = {
            "epoch": self.epoch,
            "step_in_epoch": self.step_in_epoch,
            "global_step": self.global_step,
        }

        for metric in self.metrics:
            manifest_data[metric.name.lower()] = metric.report_value

        base_path = self._get_output_path('checkpoints', directory)
        os.makedirs(base_path, exist_ok=True)

        with open(os.path.join(base_path, 'manifest.json'), 'w') as f:
            json.dump(manifest_data, f, indent=4)

        self._checkpoint_model(os.path.join(base_path, "checkpoint.pth"))

        if self.export_options.export:
            self._export_model(self.base_model, directory, 'train')
            self._export_model(self.ema_base_model, directory, 'ema')

    def _checkpoint_model(self, path: str) -> None:

        checkpoint = {
            'global_step': self.global_step,
            'step_in_epoch': self.step_in_epoch,
            'epoch': self.epoch,
            'model_state_dict': self.base_model.state_dict,
            'ema_model_state_dict': self.ema_base_model.state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }

        torch.save(checkpoint, path)

    def _export_model(self, model: Model, directory: str, prefix: str):
        fp32_path = self._get_output_path('checkpoints', directory, f"export_{prefix}_fp32.onnx")
        fp16_path = self._get_output_path('checkpoints', directory, f"export_{prefix}_fp16.onnx")
        int8_path = self._get_output_path('checkpoints', directory, f"export_{prefix}_int8.onnx")

        export_model = self.get_export_model(model)
        cpu_device = torch.device("cpu")
        export_model_fp32 = export_model.clone(device=cpu_device, dtype=torch.float32)

        export_onnx(export_model_fp32,
                    fp32_path,
                    cpu_device,
                    torch.float32,
                    self.export_options.optimize)

        if self.export_options.quantize_fp16:
            convert_fp16(fp32_path, fp16_path)

        if self.export_options.quantize_int8:
            convert_int8_dynamic(fp32_path, int8_path)

    def _explain_compilation(self, input_batch):
        if self.train_model.is_compiled and self.training_options.compile_explain:
            explanation = self.train_model.get_compilation_diagnostics(input_batch)

            output_path = self._get_output_path('compilation_report.txt')
            with open(output_path, 'w') as f:
                f.write(str(explanation))

    def _update_metrics(self, report_writer, loss_value, prediction_batch, truth_batch):
        for metric in self.metrics:
            if isinstance(metric, CalculatedMetric):
                metric.update(truth_batch, prediction_batch)
        self.loss_metric.set_current_value(loss_value)

        total_memory, free_memory = get_memory_info(self.device)
        self.free_memory_metric.update_value(free_memory / 1024 / 1024)  # MB

        report_writer.update(self.epoch, self.step_in_epoch, self.global_step, self.metrics)

    def _get_output_path(self, *args):
        return os.path.join(self.training_options.output_path, self.run_id, *args)

    def _reset_metrics(self, reset_epoch: bool, reset_report: bool):
        for metric in self.metrics:
            metric.reset(reset_epoch=reset_epoch, reset_report=reset_report)

