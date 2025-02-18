import logging
import os
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, TypedDict

import torch
from torch import GradScaler, nn, optim
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
    """
    Attributes:
        model_path: relative path from the checkpoint file to the model file
    """

    model_path: str
    optimizer_state_dict: dict[str, Any]
    lr_scheduler_state_dict: dict[str, Any]
    scaler_state_dict: dict[str, Any]
    job_metrics: dict[str, dict[str, list[float]]]


# name of the jobs
TRAIN = "train"
VAL = "val"


@dataclass
class Trainer:
    """Repeatedly run training and validation loop, log results and save checkpoints

    Example usage:
    ```
        trainer = Trainer(*params)
        if checkpoint_file is not None:
            trainer.load_checkpoint(checkpoint_file)
        trainer.train()
    ```
    """

    # --- components
    model: nn.Module
    train_loader: data.DataLoader
    train_augment: v2.Transform
    val_loader: data.DataLoader
    val_augment: v2.Transform
    criterion: nn.Module
    optimizer: optim.Optimizer
    lr_scheduler: optim.lr_scheduler.LRScheduler
    scaler: GradScaler
    device: str
    learn_step: int
    num_epochs: int
    loss_weight: dict[str, float]
    num_classes: int
    # --- util
    labels: Sequence[str]
    colors: Sequence[tuple[int, int, int]]
    out_folder: Path | None
    checkpoint_epochs: int = 1
    best_by: str = "max:miou"
    """In the form of `"[max|min]:[metric]"` where metric must be a valid key in metrics"""
    loggers: Sequence[Logger] = ()
    num_snapshot_data = 4

    def __post_init__(self):
        if len(self.labels) != self.num_classes:
            raise ValueError(f"Labels have different size than num_classes")
        if len(self.colors) != self.num_classes:
            raise ValueError(f"Colors have different size than num_classes")

        self.job_metrics: dict[str, dict[str, list[float]]] = {TRAIN: {}, VAL: {}}
        self.model.to(self.device)
        self.criterion.to(self.device)

    def train(self):
        with ExitStack() as stack:
            [stack.enter_context(logger) for logger in self.loggers]

            start_epoch = 0
            if len(self.job_metrics[TRAIN]) > 0:
                start_epoch = len(next(iter(self.job_metrics[TRAIN].values())))

            for i in range(start_epoch, self.num_epochs):
                logger.info(f"----- Epoch [{i:>4}/{self.num_epochs}] -----")
                train_ms = engine.train_one_epoch(
                    data_loader=self.train_loader,
                    augment=self.train_augment,
                    desc=TRAIN,
                    **self.__dict__,
                )
                self.lr_scheduler.step()
                self.record_metrics(TRAIN, i, train_ms)
                self.save_snapshot(TRAIN, i, self.train_loader.dataset)

                val_ms = engine.eval_one_epoch(
                    data_loader=self.val_loader,
                    augment=self.val_augment,
                    desc=VAL,
                    **self.__dict__,
                )
                self.record_metrics(VAL, i, val_ms)
                self.save_snapshot(VAL, i, self.val_loader.dataset)

                if self.out_folder is None:
                    continue
                if (i + 1) % self.checkpoint_epochs == 0:
                    model_file = self.out_folder / "model" / f"e{i:>04}.pth"
                    checkpoint_file = self.out_folder / "checkpoint" / f"e{i:>04}.pth"
                    self.save_checkpoint(model_file, checkpoint_file)
                    logger.info(f"Checkpoint saved to {checkpoint_file}")

                # always save latest checkpoint and model
                model_file = self.out_folder / f"latest_model.pth"
                checkpoint_file = self.out_folder / "latest_checkpoint.pth"
                self.save_checkpoint(model_file, checkpoint_file)
                logger.debug(f"Latest checkpoint saved to {checkpoint_file}")

                # TODO save the best model

            logger.info(f"Training completed")

    def record_metrics(self, job: str, step: int, ms: MetricStore):
        for l in self.loggers:
            l.log_epoch_metrics(job, step, ms)
        metrics = ms.summarize()
        metrics_text = "| ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.debug(f"Metrics for {job} {step}: {metrics_text}")

        for k, v in metrics.items():
            self.job_metrics[job].setdefault(k, [])
            self.job_metrics[job][k].append(v)
        for l in self.loggers:
            l.log_running_metrics(self.job_metrics)

    def save_snapshot(self, job: str, step: int, dataset: data.Dataset):
        snapshot = engine.create_snapshot(
            dataset=dataset,
            augment=self.val_augment,
            num_samples=self.num_snapshot_data,
            **self.__dict__,
        )
        for l in self.loggers:
            l.save_snapshot(job, step, snapshot)

    def save_checkpoint(self, model_file: Path, checkpoint_file: Path):
        """save model in separate file for inference"""
        model_file.parent.mkdir(exist_ok=True)
        checkpoint_file.parent.mkdir(exist_ok=True)
        torch.save(self.model.state_dict(), model_file)
        checkpoint: Checkpoint = {
            "model_path": str(os.path.relpath(model_file, start=checkpoint_file)),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "job_metrics": self.job_metrics,
        }
        torch.save(checkpoint, checkpoint_file)

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
