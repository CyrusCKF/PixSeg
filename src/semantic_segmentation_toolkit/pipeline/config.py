import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import GradScaler, Tensor, nn
from torch.hub import load_state_dict_from_url
from torch.nn.modules.loss import _Loss as Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils import data
from torchvision.transforms import v2

from ..datasets import DATASET_ZOO, DatasetMeta, resolve_metadata
from ..learn import CLASS_WEIGHTINGS, CRITERION_ZOO, LR_SCHEDULER_ZOO, OPTIMIZER_ZOO
from ..models import MODEL_WEIGHTS, MODEL_ZOO
from ..utils.transform import SegmentationAugment, SegmentationTransform
from .logger import LocalLogger, Logger, TensorboardLogger, WandbLogger
from .trainer import Trainer

logger = logging.getLogger(__name__)


class Config:
    """Please check `doc/config.md` for explanation of each fields"""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._dataset_meta: DatasetMeta | None = None
        self._out_folder: Path | None = None

    @property
    def dataset_meta(self) -> DatasetMeta:
        if self._dataset_meta is None:
            key = self.config["data"]["dataset"]["dataset"]
            self._dataset_meta = resolve_metadata(key)
        return self._dataset_meta

    @property
    def checkpoint_file(self) -> Path | None:
        checkpoint_file = self.config["paths"].get("checkpoint")
        if checkpoint_file is None:
            return None
        return Path(checkpoint_file)

    @property
    def out_folder(self) -> Path:
        """Generated subfolder to log the run"""
        if self._out_folder is None:
            runs_folder = self.config["paths"]["runs_folder"]
            sub_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._out_folder = Path(runs_folder) / sub_folder
        return self._out_folder

    @property
    def device(self) -> str:
        device = self.config["trainer"]["device"]
        if device != "auto":
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def build_model(self) -> nn.Module:
        model_name = self.config["model"]["model"]
        params: dict = self.config["model"]["params"].copy()
        weights = params.pop("weights", None)
        num_classes = self.dataset_meta.num_classes
        model = MODEL_ZOO[model_name](num_classes=num_classes, **params)

        # safely load model state dict
        state_dict = None
        state_file = self.config["model"].get("state_file")
        if weights is not None and state_file is not None:
            raise ValueError("Expect at most one of state_file or params.weights")
        # access weights state dict directly if possible
        if weights is not None:
            seg_weights = None
            if model_name in MODEL_WEIGHTS:
                seg_weights = MODEL_WEIGHTS[model_name].resolve(weights)

            if seg_weights is not None:
                state_dict = load_state_dict_from_url(
                    seg_weights.url, progress=params.get("progress", True)
                )
            else:
                model_with_weights = MODEL_ZOO[model_name](weights=weights, **params)
                state_dict = model_with_weights.state_dict()
        if state_file is not None:
            state_dict = torch.load(state_file)

        if state_dict is not None:
            safe_transfer_state_dict(model, state_dict)
        return model

    def build_datasets(self) -> tuple[data.Dataset, data.Dataset]:
        ignore_index = self.dataset_meta.ignore_index
        train_size = self.config["data"]["dataset"]["pad_crop_size"]
        if train_size == "none":
            train_size = None
        train_transform = SegmentationTransform(train_size, mask_fill=ignore_index)
        val_transform = SegmentationTransform(mask_fill=ignore_index)

        entry = DATASET_ZOO[self.config["data"]["dataset"]["dataset"]]
        params = self.config["data"]["dataset"]["params"]
        train_dataset = entry.construct_train(transforms=train_transform, **params)
        val_dataset = entry.construct_val(transforms=val_transform, **params)
        return train_dataset, val_dataset

    def build_data_loaders(
        self, train_dataset: data.Dataset, val_dataset: data.Dataset
    ) -> tuple[data.DataLoader, data.DataLoader]:
        num_workers = self.config["data"]["loader"]["num_workers"]
        train_params = self.config["data"]["loader"]["params"]
        train_loader = data.DataLoader(
            train_dataset, num_workers=num_workers, **train_params
        )
        val_loader = data.DataLoader(val_dataset, num_workers=num_workers)
        return train_loader, val_loader

    def build_data_augments(self) -> tuple[v2.Transform, v2.Transform]:
        ignore_index = self.dataset_meta.ignore_index
        train_params = self.config["data"]["augment"]["params"]
        train_augment = SegmentationAugment(**train_params, mask_fill=ignore_index)
        val_augment = SegmentationAugment(mask_fill=ignore_index)
        return train_augment, val_augment

    def build_criterion(self, dataset: data.Dataset) -> Loss:
        meta = self.dataset_meta
        weighting = self.config["criterion"]["class_weight"]
        if isinstance(weighting, list):
            weight = torch.tensor(weighting)
        else:
            weighting = self.config["criterion"]["class_weight"]
            weight = CLASS_WEIGHTINGS[weighting](dataset, meta.num_classes)

        crit = self.config["criterion"]["criterion"]
        params = self.config["criterion"]["params"]
        return CRITERION_ZOO[crit](
            ignore_index=meta.ignore_index, weight=weight, **params
        )

    def build_optimizer(self, model: nn.Module) -> Optimizer:
        optim = self.config["optimizer"]["optimizer"]
        params = self.config["optimizer"]["params"]
        return OPTIMIZER_ZOO[optim](model.parameters(), **params)

    def build_lr_scheduler(self, optimizer: Optimizer) -> LRScheduler:
        lr = self.config["lr_scheduler"]["lr_scheduler"]
        params = self.config["lr_scheduler"]["params"]
        return LR_SCHEDULER_ZOO[lr](optimizer, **params)

    def build_scaler(self) -> GradScaler:
        params = self.config["scaler"]["params"]
        return GradScaler(self.device, **params)

    def get_trainer_params(self) -> dict[str, Any]:
        params = self.config["trainer"]["params"]
        params["out_folder"] = self.out_folder
        params["device"] = self.device

        aux_weight = self.config["criterion"]["aux_weight"]
        has_aux_loss = self.config["model"]["params"].get("aux_loss", False)
        if aux_weight != 0 and not has_aux_loss:
            warnings.warn(
                "aux_weight is set to non-zero while aux_loss for model is not detected."
                " This may not have any effect."
            )
        params["loss_weight"] = {"aux": aux_weight}

        batch_size = self.config["data"]["loader"]["params"].get("batch_size", 1)
        effective_batch_size = self.config["optimizer"]["effective_batch_size"]
        if effective_batch_size % batch_size != 0:
            raise ValueError(
                "effective_batch_size must be a multiple of batch_size,"
                f" but got {effective_batch_size=}, {batch_size=}"
            )
        params["learn_step"] = effective_batch_size // batch_size

        return params

    def build_loggers(self) -> list[Logger]:
        loggers: list[Logger] = [LocalLogger(self.out_folder, self.dataset_meta.labels)]
        config_to_log = {k: v for k, v in self.config.items() if k != "log"}
        config_dict = _flatten_nested_dict(config_to_log)

        wandb_table = self.config["log"]["wandb"]
        wandb_key = wandb_table.get("api_key")
        if wandb_key is not None:
            run_id = wandb_table.get("run_id")
            wandb_params = wandb_table["params"]
            wandb_logger = WandbLogger(
                wandb_key, run_id, config=config_dict.copy(), **wandb_params
            )
            loggers.append(wandb_logger)

        tensorboard_table = self.config["log"]["tensorboard"]
        tensorboard_enabled = tensorboard_table["enabled"]
        if tensorboard_enabled:
            tensorboard_logger = TensorboardLogger(
                config=config_dict.copy(), **tensorboard_table["params"]
            )
            loggers.append(tensorboard_logger)

        return loggers

    def to_trainer(self) -> Trainer:
        model = self.build_model()
        train_dataset, val_dataset = self.build_datasets()
        train_loader, val_loader = self.build_data_loaders(train_dataset, val_dataset)
        train_augment, val_augment = self.build_data_augments()
        criterion = self.build_criterion(train_dataset)
        optimizer = self.build_optimizer(model)
        lr_scheduler = self.build_lr_scheduler(optimizer)
        scaler = self.build_scaler()
        loggers = self.build_loggers()

        trainer_kwargs = self.get_trainer_params()
        dataset_meta_kwargs = self.dataset_meta.__dict__.copy()
        dataset_meta_kwargs.pop("ignore_index")

        # fmt: off
        trainer = Trainer(
            model, train_loader, train_augment, val_loader, val_augment, criterion, optimizer, 
            lr_scheduler, scaler, loggers=loggers, **trainer_kwargs, **dataset_meta_kwargs
        )
        # fmt: on
        if self.checkpoint_file is not None:
            trainer.load_checkpoint(self.checkpoint_file)
        return trainer


def _flatten_nested_dict(nested_dict: dict[str, Any], sep="/") -> dict[str, Any]:
    """Flatten dict by only keeping its last keys while ensuring it is
    still uniquely identifiable

    Example:
    ```
    { "A": { "AA": 1 }, "B": { "BB": { "DUP": 2 } }, "DUP": 3 }
    => { "AA": 1, "BB/DUP": 2, "DUP": 3 }
    ```
    """

    def flatten(
        dict_: dict[str, Any], parent_key: tuple[str, ...] = ()
    ) -> dict[tuple[str, ...], Any]:
        flattened: dict[tuple[str, ...], Any] = {}
        for k, v in dict_.items():
            new_key = parent_key + (k,)
            if isinstance(v, dict):
                flattened.update(flatten(v, new_key))
            else:
                flattened[new_key] = v
        return flattened

    flattened_dict = flatten(nested_dict)

    result = {}
    for full_key, value in flattened_dict.items():
        # try all shortened key
        for i in range(1, len(full_key) + 1):
            short_key = full_key[-i:]
            other_keys = {k[-i:] for k in flattened_dict.keys() if k != full_key}
            if short_key in other_keys:
                continue

            joined = sep.join(short_key)
            result[joined] = value
            break
    return result


def safe_transfer_state_dict(model: nn.Module, state_dict: dict[str, Tensor]):
    logger.info("Transferring weights to model ...")
    filtered_state_dict = {}
    mismatch_keys = []
    for k, v in state_dict.items():
        model_v = model.state_dict().get(k, None)
        if isinstance(model_v, Tensor) and model_v.shape != v.shape:
            mismatch_keys.append(k)
        else:
            filtered_state_dict[k] = v

    missing_keys, unexpected_keys = model.load_state_dict(
        filtered_state_dict, strict=False
    )
    if not missing_keys and not unexpected_keys and len(mismatch_keys) == 0:
        logger.info("All weights are transferred successfully")
    else:
        logger.info(
            f"Transfer completed. Found abnormal keys:"
            f" {missing_keys=}, {unexpected_keys=}, {mismatch_keys=}"
        )
