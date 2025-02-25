import sys
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss, _WeightedLoss

sys.path.append(str((Path(__file__) / "..").resolve()))
from learn_api import CRITERION_ZOO


def register_criterion(name: str | None = None):
    def wrapper(callable: Callable[..., _Loss]) -> Callable[..., _Loss]:
        key = callable.__name__ if name is None else name
        if key in CRITERION_ZOO:
            raise KeyError(f"An entry is already registered under the key '{key}'.")
        CRITERION_ZOO[key] = callable
        return callable

    return wrapper


register_criterion()(CrossEntropyLoss)
# TODO make Focal loss


# TODO test this
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
        super().__init__(weight, None, None, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.eps = 1e-6

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute the Dice loss"""
        input = F.softmax(input, dim=1)
        target_one_hot = F.one_hot(target, num_classes=input.shape[1]).float()

        mask = target != self.ignore_index
        input = input * mask.unsqueeze(1)
        target_one_hot = target_one_hot * mask.unsqueeze(1)

        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            target_one_hot = (
                target_one_hot * (1 - self.label_smoothing)
                + self.label_smoothing / input.shape[1]
            )

        # Compute intersection and union
        intersection = input * target_one_hot.sum(dim=(0, 1))
        union = input.sum(dim=(0, 1)) + target_one_hot.sum(dim=(0, 1))

        # Compute Dice score
        dice_score = (2 * intersection + self.eps) / (union + self.eps)
        dice_loss = 1 - dice_score

        # Apply class weights if specified
        if self.weight is not None:
            dice_loss = dice_loss * self.weight

        # Reduce the loss
        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


def _test():
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss()

    x = torch.tensor([[1, 2, 1], [1, 1, 2]], dtype=torch.float)
    y = torch.tensor([1, 1])
    print(ce_loss(x, y), dice_loss(x, y))


if __name__ == "__main__":
    _test()
