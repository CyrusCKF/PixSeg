import sys
import warnings
from pathlib import Path
from typing import Callable, ParamSpec

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss, _WeightedLoss

sys.path.append(str((Path(__file__) / "..").resolve()))
from learn_api import CRITERION_ZOO

P = ParamSpec("P")


def register_criterion(name: str | None = None):
    def wrapper(callable: Callable[P, _Loss]) -> Callable[P, _Loss]:
        key = callable.__name__ if name is None else name
        if key in CRITERION_ZOO:
            raise KeyError(f"An entry is already registered under the key '{key}'.")
        CRITERION_ZOO[key] = callable
        return callable

    return wrapper


register_criterion()(CrossEntropyLoss)


@register_criterion()
class DiceLoss(_WeightedLoss):
    """Implement dice loss from [Generalised Dice overlap as a deep learning loss
    function for highly unbalanced segmentations](https://arxiv.org/abs/1707.03237)

    Roughly speaking, `DiceLoss(y, p) = 1 - (2yp + 1) / (y + p + 1)`

    The *input* is expected to contain the unnormalized logits for each class
    (which do *not* need to be positive or sum to 1, in general).

    *input* should be float Tensor of size (batch_size, num_classes, d1, d2, ..., dk).

    *target* should be int Tensor of size (batch_size, d1, d2, ..., dk) where each
        value should be between [0, num_classes)
    """

    def __init__(
        self,
        weight: Tensor | None = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        """See :class:`CrossEntropyLoss` for each argument"""
        if weight is not None:
            weight /= weight.sum()
        super().__init__(weight, None, None, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.eps = 1e-6

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        num_classes = input.size(1)
        input = F.softmax(input, dim=1)
        target_one_hot = torch.stack([target == i for i in range(num_classes)], dim=1)
        target_one_hot = target_one_hot.to(torch.float)

        if self.label_smoothing > 0:
            target_one_hot *= 1 - self.label_smoothing
            target_one_hot += self.label_smoothing / num_classes

        mask = target != self.ignore_index
        input = input * mask.unsqueeze(1)
        target_one_hot = target_one_hot * mask.unsqueeze(1)

        # Compute dice
        intersection = input * target_one_hot
        union = input + target_one_hot
        dice_score = (2 * intersection + self.eps) / (union + self.eps)
        dice_loss = 1 - dice_score

        if self.weight is not None:
            assert len(self.weight) == num_classes
            # multiply along the N dim in [B, N, d1, d2, ..., dk] of target_one_hot
            shape = [1, -1] + [1] * (target_one_hot.dim() - 2)
            target_one_hot *= self.weight.view(shape)
        dice_loss = dice_loss.mean(dim=1)

        # Reduce the loss
        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


@register_criterion()
class FocalLoss(_WeightedLoss):
    """Implement focal loss from [Focal Loss for Dense Object
    Detection](https://arxiv.org/pdf/1708.02002)

    Mathematically, `FocalLoss(pt) = -(1 - pt) ^ gamma * log(pt)` where `-log(pt)`
    is the cross entropy loss.

    *input* and *target* follows from :class:`CrossEntropyLoss`
    """

    def __init__(
        self,
        weight: Tensor | None = None,
        gamma: float = 1,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        """See :class:`CrossEntropyLoss` for each argument

        Args:
            gamma: The focusing parameter for focal loss. When gamma is `0`, this is
                in theory the same as :class:`CrossEntropyLoss`. But due to weight
                normalization for :args:`weight` or :args:`ignore_index`, the value
                can be different.
        """
        if weight is not None:
            weight /= weight.sum()
        super().__init__(weight, None, None, reduction)
        self.gamma = gamma
        if self.gamma == 0:
            warnings.warn(
                "Input gamma is 0. This will make the behaviour same as"
                " cross entropy loss"
            )
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        ce = F.cross_entropy(
            input,
            target,
            self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = (-ce).exp()
        loss = (1 - pt).pow(self.gamma) * ce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def _test():
    num_classes = 20
    ce_cri = CrossEntropyLoss(
        ignore_index=num_classes,
    )
    dice_cri = DiceLoss(
        weight=torch.rand([num_classes]),
        ignore_index=num_classes,
        label_smoothing=0.1,
    )
    focal_cri = FocalLoss(
        weight=torch.rand([num_classes]),
        ignore_index=num_classes,
        label_smoothing=0.1,
        gamma=0.1,
    )

    logits = torch.rand([4, num_classes, 160, 90]) * 5 - 2
    masks = torch.randint(0, num_classes + 1, [4, 160, 90])

    ce_loss = ce_cri(logits, masks)
    dice_loss = dice_cri(logits, masks)
    focal_loss = focal_cri(logits, masks)
    print(ce_loss, dice_loss, focal_loss)
    print(ce_loss.shape, dice_loss.shape, focal_loss.shape)


if __name__ == "__main__":
    _test()
