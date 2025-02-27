import random
from timeit import default_timer
from typing import Sequence

import torch
import tqdm
from torch import GradScaler, Tensor, nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils import data
from torchvision.transforms import v2

from ..utils.metrics import MetricStore
from ..utils.visual import draw_mask_on_image


def forward_batch(
    model: nn.Module,
    images: Tensor,
    masks: Tensor | None,
    augment: v2.Transform,
    criterion: nn.Module | None,
    device: str,
    **kwargs,
):
    """Return a tuple of (logits, losses)

    losses will be default if criterion is `None`

    :param:`model` and :param:`criterion` are assumed to be on :param:`device`
    """
    images = images.to(device)
    if masks is None:
        masks = torch.zeros_like(images, dtype=torch.long)
    masks = masks.to(device)
    images, masks = augment(images, masks)

    with torch.autocast(device_type=device, enabled=device != "cpu"):
        logits: dict[str, Tensor] = model(images)
        mask_size = masks.shape[-2:]  # type: ignore
        for k, v in logits.items():
            logits[k] = F.interpolate(v, mask_size, mode="bilinear")

        losses: dict[str, Tensor] = {}
        if criterion is not None:
            losses = {k: criterion(v, masks) for k, v in logits.items()}
    return logits, losses


def train_one_epoch(
    model: nn.Module,
    data_loader: data.DataLoader,
    augment: v2.Transform,
    criterion: nn.Module,
    optimizer: Optimizer,
    scaler: GradScaler,
    device: str,
    learn_step: int,
    num_classes: int,
    loss_weight: dict[str, float],
    desc="Train",
    silent=False,
    **kwargs,
) -> MetricStore:
    """Train the given model for one epoch

    :param:`model` and :param:`criterion` are assumed to be on :param:`device`

    Args:
        learn_step: Number of iterations before back propagating.
            Used to increase effective batch size. Set to `1` for normal optimization loop.
            Set to `0` for full batch learning. If batch size in :param:`dataloader` is `1`,
            this is the same as effective batch size.
        loss_weight: Weight for each named loss. Mainly used for "out" and "aux"
    """
    model.train()
    ms = MetricStore(num_classes)
    loader = tqdm.tqdm(
        enumerate(data_loader), total=len(data_loader), desc=desc, disable=silent
    )
    for i, (images, masks) in loader:
        start_time = default_timer()
        logits, losses = forward_batch(model, images, masks, augment, criterion, device)
        loss_sum = sum(v * loss_weight.get(k, 1) for k, v in losses.items())

        if isinstance(loss_sum, Tensor):
            scaler.scale(loss_sum).backward()
        if (i + 1) % learn_step == 0 or i == len(loader) - 1:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        end_time = default_timer()

        preds = logits["out"].argmax(1)
        ms.store_results(masks, preds)
        batch_size = images.size(0)
        measures = {
            "loss": losses["out"].item() * batch_size,
            "time": end_time - start_time,
        }
        ms.store_measures(batch_size, measures)
        if not silent:
            loader.set_postfix(ms.summarize())

    return ms


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    data_loader: data.DataLoader,
    augment: v2.Transform,
    criterion: nn.Module,
    device: str,
    num_classes: int,
    desc="Eval",
    silent=False,
    **kwargs,
) -> MetricStore:
    """Evaluate the given model for one epoch

    :param:`model` and :param:`criterion` are assumed to be on :param:`device`
    """
    model.eval()
    ms = MetricStore(num_classes)
    loader = tqdm.tqdm(
        iter(data_loader), total=len(data_loader), desc=desc, disable=silent
    )
    for images, masks in loader:
        start_time = default_timer()
        logits, losses = forward_batch(model, images, masks, augment, criterion, device)
        end_time = default_timer()

        preds = logits["out"].argmax(1)
        ms.store_results(masks, preds)
        batch_size = images.size(0)
        measures = {
            "loss": losses["out"].item() * batch_size,
            "time": end_time - start_time,
        }
        ms.store_measures(batch_size, measures)
        if not silent:
            loader.set_postfix(ms.summarize())

    return ms


@torch.no_grad()
def create_snapshots(
    model: nn.Module,
    dataset: data.Dataset[tuple[Tensor, Tensor]],
    augment: v2.Transform,
    device: str,
    colors: Sequence[tuple[int, int, int]],
    num_data: int = 1,
    **kwargs,
) -> list[list[Tensor]]:
    """Return list of images in sets of three: original, ground truth overlay,
    and prediction overlay.

    :param:`model` is assumed to be on :param:`device`
    """
    model.eval()
    snapshots: list[list[Tensor]] = []
    image_indices = random.sample(range(len(dataset)), num_data)  # type: ignore
    for i in image_indices:
        image, mask = dataset[i]
        images, masks = image.unsqueeze(0), mask.unsqueeze(0)
        logits, _ = forward_batch(model, images, masks, augment, None, device)
        preds = torch.argmax(logits["out"], 1).to(torch.long)

        image = images[0]
        mask_overlay = draw_mask_on_image(image, masks[0], colors, (255, 255, 255))
        pred_overlay = draw_mask_on_image(image, preds[0], colors)
        snapshots.append([image, mask_overlay, pred_overlay])
    return snapshots
