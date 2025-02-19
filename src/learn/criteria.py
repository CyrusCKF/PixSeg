import sys
from pathlib import Path
from typing import Callable

from torch.nn import CrossEntropyLoss
from torch.nn.modules.loss import _Loss

sys.path.append(str((Path(__file__) / "..").resolve()))
from learning_zoo import CRITERION_ZOO


def register_criterion(callable: Callable[..., _Loss]) -> Callable[..., _Loss]:
    key = callable.__name__
    if key in CRITERION_ZOO:
        raise ValueError(f"An entry is already registered under the name '{key}'.")
    CRITERION_ZOO[key] = callable
    return callable


register_criterion(CrossEntropyLoss)
# TODO make Dice Loss, Focal loss
