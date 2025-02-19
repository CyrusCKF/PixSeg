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
from torch.nn.modules.loss import _Loss as Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils import data
from torchvision.transforms import v2

sys.path.append(str((Path(__file__) / "..").resolve()))
from logger import Logger
from trainer import Trainer

sys.path.append(str((Path(__file__) / "../../..").resolve()))
from src.datasets import DATASET_METADATA, DATASET_ZOO, DatasetMeta
from src.learn import CLASS_WEIGHTINGS, CRITERION_ZOO, LR_SCHEDULER_ZOO, OPTIMIZER_ZOO
from src.models import MODEL_ZOO

logger = logging.getLogger(__name__)


class Config:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._dataset_meta: DatasetMeta | None = None

    def build_model(self) -> nn.Module:
        num_classes = self.dataset_meta.num_classes
        return MODEL_ZOO[self.config["model"]["model"]](
            num_classes=num_classes, **self.config["model"]["params"]
        )

    def build_datasets(self) -> tuple[data.Dataset, data.Dataset]:
        raise NotImplementedError()

    @property
    def dataset_meta(self) -> DatasetMeta:
        if self._dataset_meta is None:
            self._dataset_meta = DATASET_ZOO[
                self.config["data"]["dataset"]["dataset"]
            ].meta
        return self._dataset_meta

    def build_data_loaders(self) -> tuple[data.DataLoader, data.DataLoader]:
        raise NotImplementedError()

    def build_data_augments(self) -> tuple[v2.Transform, v2.Transform]:
        raise NotImplementedError()

    def build_criterion(self) -> Loss:
        raise NotImplementedError()

    def build_optimizer(self, model: nn.Module) -> Optimizer:
        raise NotImplementedError()

    def build_lr_scheduler(self, optimizer: Optimizer) -> LRScheduler:
        raise NotImplementedError()

    def get_trainer_params(self) -> dict[str, Any]:
        raise NotImplementedError()

    def build_loggers(self) -> list[Logger]:
        raise NotImplementedError()

    @property
    def checkpoint_file(self) -> Path | None:
        raise NotImplementedError()

    @property
    def run_folder(self) -> Path:
        """Generated subfolder to log the run"""
        raise NotImplementedError()

    def to_trainer(self) -> Trainer:
        model = self.build_model()
        train_loader, val_loader = self.build_data_loaders()
        train_augment, val_augment = self.build_data_augments()
        criterion = self.build_criterion()
        optimizer = self.build_optimizer(model)
        lr_scheduler = self.build_lr_scheduler(optimizer)
        loggers = self.build_loggers()

        trainer_kwargs = self.get_trainer_params()
        dataset_meta_kwargs = self.dataset_meta.__dict__
        dataset_meta_kwargs.pop("ignore_index")

        # fmt: off
        trainer = Trainer(
            model, train_loader, train_augment, val_loader, val_augment, criterion, optimizer, 
            lr_scheduler, GradScaler(), loggers=loggers, **trainer_kwargs, **dataset_meta_kwargs
        )
        # fmt: on
        if self.checkpoint_file is not None:
            trainer.load_checkpoint(self.checkpoint_file)
        return trainer


def _test():
    import toml

    config_toml = toml.load(r"doc\sample_config.toml")
    config = Config(config_toml)
    trainer = config.to_trainer()


if __name__ == "__main__":
    _test()
