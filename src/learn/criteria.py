import sys
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar

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
# TODO make Focal loss


@register_criterion()
class DiceLoss(_WeightedLoss):
    """Dice loss for multi class.

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
        """Compute the Dice loss"""
        num_classes = input.size(1)
        input = F.softmax(input, dim=1)
        target_one_hot = torch.stack([target == i for i in range(num_classes)], dim=1)
        target_one_hot = target_one_hot.to(torch.float)

        mask = target != self.ignore_index
        input = input * mask.unsqueeze(1)
        target_one_hot = target_one_hot * mask.unsqueeze(1)

        if self.label_smoothing > 0:
            target_one_hot *= 1 - self.label_smoothing
            target_one_hot += self.label_smoothing / num_classes

        # Compute dice
        spatial_dims = list(range(2, input.dim()))
        intersection = (input * target_one_hot).sum(dim=spatial_dims)
        union = input.sum(dim=spatial_dims) + target_one_hot.sum(dim=spatial_dims)
        dice_score = (2 * intersection + self.eps) / (union + self.eps)
        dice_loss = 1 - dice_score

        if self.weight is not None:
            assert len(self.weight) == num_classes
            dice_loss *= self.weight
        dice_loss = dice_loss.mean(dim=1)

        # Reduce the loss
        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


def _test():
    num_classes = 20
    ce_loss = CrossEntropyLoss(
        ignore_index=num_classes,
    )
    dice_loss = DiceLoss(
        # weight=torch.rand([num_classes]),
        ignore_index=num_classes,
        label_smoothing=0.1,
        reduction="mean",
    )

    logits = torch.rand([4, num_classes, 160, 90]) * 5 - 2
    masks = torch.randint(0, num_classes + 1, [4, 160, 90])

    print(ce_loss(logits, masks), dice_loss(logits, masks))


if __name__ == "__main__":
    _test()
