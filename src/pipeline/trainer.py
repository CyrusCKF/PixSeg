import logging
import os
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, TypedDict

import numpy as np
import torch
from torch import GradScaler, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils import data
from torchvision.transforms import v2

sys.path.append(str((Path(__file__) / "..").resolve()))
import engine
from logger import Logger

sys.path.append(str((Path(__file__) / "../../..").resolve()))
from src.utils.metrics import MetricStore

logger = logging.getLogger(__name__)


# use TypedDict for easier serialization
class Checkpoint(TypedDict):
    """Checkpoint for restoring training. Model state is saved in a separate file.

    Attributes:
        model_path: relative path from the checkpoint file to the model file
    """

    model_path: str
    optimizer_state_dict: dict[str, Any]
    lr_scheduler_state_dict: dict[str, Any]
    scaler_state_dict: dict[str, Any]
    job_metrics: dict[str, dict[str, list[float]]]


@dataclass
class Trainer:
    """Repeatedly run training and validation loop, provide detailed logs
    and save checkpoints

    Example usage:
    ```
        trainer = Trainer(*params)
        if checkpoint_file is not None:
            trainer.load_checkpoint(checkpoint_file)
        trainer.train()
    ```
    """

    # name of the jobs
    TRAIN = "train"
    VAL = "val"

    # --- components
    model: nn.Module
    train_loader: data.DataLoader
    train_augment: v2.Transform
    val_loader: data.DataLoader
    val_augment: v2.Transform
    criterion: nn.Module
    optimizer: Optimizer
    lr_scheduler: LRScheduler
    scaler: GradScaler
    device: str
    learn_step: int
    num_epochs: int
    num_classes: int
    loss_weight: dict[str, float]
    # util
    labels: Sequence[str]
    colors: Sequence[tuple[int, int, int]]
    out_folder: Path | None
    checkpoint_steps: int = 1
    best_by: str = "max:miou"
    """In the form of `"[max|min]:[metric]"` where metric must be a valid key in metrics"""
    loggers: Sequence[Logger] = ()
    num_snapshots: int = 4

    def __post_init__(self):
        if len(self.labels) != self.num_classes:
            raise ValueError(f"Labels have different size than num_classes")
        if len(self.colors) != self.num_classes:
            raise ValueError(f"Colors have different size than num_classes")

        self.job_metrics: dict[str, dict[str, list[float]]] = {
            self.TRAIN: {},
            self.VAL: {},
        }
        self.model.to(self.device)
        self.criterion.to(self.device)

    def train(self):
        with ExitStack() as stack:
            [stack.enter_context(logger) for logger in self.loggers]

            start_epoch = 0
            if len(self.job_metrics[self.TRAIN]) > 0:
                start_epoch = len(next(iter(self.job_metrics[self.TRAIN].values())))
            for i in range(start_epoch, self.num_epochs):
                self.run_one_epoch(i)

            logger.info(f"Training completed")

    def run_one_epoch(self, step: int):
        logger.info(f"----- Epoch [{step:>4}/{self.num_epochs}] -----")
        train_ms = engine.train_one_epoch(
            data_loader=self.train_loader,
            augment=self.train_augment,
            desc=self.TRAIN,
            **self.__dict__,
        )
        self.lr_scheduler.step()
        self.record_metrics(self.TRAIN, step, train_ms)
        self.save_snapshot(self.TRAIN, step, self.train_loader.dataset)

        val_ms = engine.eval_one_epoch(
            data_loader=self.val_loader,
            augment=self.val_augment,
            desc=self.VAL,
            **self.__dict__,
        )
        self.record_metrics(self.VAL, step, val_ms)
        self.save_snapshot(self.VAL, step, self.val_loader.dataset)
        self.export_checkpoints(step)

    def record_metrics(self, job: str, step: int, ms: MetricStore):
        metrics = ms.summarize()
        for l in self.loggers:
            l.on_job_epoch_ended(job, step, ms.confusion_matrix, metrics)
        metrics_text = "| ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.debug(f"Metrics for {job} {step}: {metrics_text}")

        for k, v in metrics.items():
            self.job_metrics[job].setdefault(k, [])
            self.job_metrics[job][k].append(v)
        for l in self.loggers:
            l.on_running_metrics_updated(self.job_metrics)

    def save_snapshot(self, job: str, step: int, dataset: data.Dataset):
        snapshots = engine.create_snapshots(
            dataset=dataset,
            augment=self.val_augment,
            num_data=self.num_snapshots,
            **self.__dict__,
        )
        for l in self.loggers:
            l.on_snapshots_created(job, step, snapshots)

    def export_checkpoints(self, step: int):
        if self.out_folder is None:
            return

        if (step + 1) % self.checkpoint_steps == 0:
            paths = _get_save_paths(self.out_folder, None, step)
            _save_checkpoint(self, *paths)
            for l in self.loggers:
                l.on_checkpoint_saved(*paths)

        # always save latest checkpoint and model
        paths = _get_save_paths(self.out_folder, "latest", None)
        _save_checkpoint(self, *paths)
        for l in self.loggers:
            l.on_checkpoint_saved(*paths)

        # save the best model
        best_index = _find_best_index(self.best_by, self.job_metrics[self.VAL])
        if best_index == step:
            logger.info("Found new best model")
            paths = _get_save_paths(self.out_folder, "best", None)
            _save_checkpoint(self, *paths)
            for l in self.loggers:
                l.on_checkpoint_saved(*paths)

    def load_checkpoint(self, checkpoint_file: Path):
        logger.info(f"Loading checkpoint in {checkpoint_file}")
        checkpoint: Checkpoint = torch.load(checkpoint_file, weights_only=True)
        model_path = checkpoint_file / checkpoint["model_path"]
        model_state_dict = torch.load(model_path, weights_only=True)

        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.job_metrics = checkpoint["job_metrics"]


def _get_save_paths(folder: Path, name: str | None, step: int | None):
    """Provide exactly one of `name` or `step`

    If step is provided, it will return path in a subfolder
    """
    if folder is None:
        return
    if name is None == step is None:
        raise ValueError("Accept exactly one of name or step")
    model_file = folder / f"{name}_model.pth"
    checkpoint_file = folder / f"{name}_checkpoint.pth"
    if step is not None:
        model_file = folder / "model" / f"e{step:>04}.pth"
        checkpoint_file = folder / "checkpoint" / f"e{step:>04}.pth"
    return model_file, checkpoint_file


def _save_checkpoint(trainer: Trainer, model_file: Path, checkpoint_file: Path):
    model_file.parent.mkdir(exist_ok=True)
    checkpoint_file.parent.mkdir(exist_ok=True)
    model_state_dict = {k: v.cpu() for k, v in trainer.model.state_dict().items()}
    torch.save(model_state_dict, model_file)

    checkpoint: Checkpoint = {
        "model_path": str(os.path.relpath(model_file, start=checkpoint_file)),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "lr_scheduler_state_dict": trainer.lr_scheduler.state_dict(),
        "scaler_state_dict": trainer.scaler.state_dict(),
        "job_metrics": trainer.job_metrics,
    }
    torch.save(checkpoint, checkpoint_file)


def _find_best_index(best_by: str, metrics_list: dict[str, list[float]]) -> int:
    """`best_by` is like [max|min]:[metric]"""
    algo_name, metric_key = best_by.split(":")
    algo = {"max": np.argmax, "min": np.argmin}[algo_name]
    metrics = metrics_list[metric_key]
    best_index = algo(metrics)
    return best_index
