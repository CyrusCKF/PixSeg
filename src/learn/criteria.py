import sys
from pathlib import Path
from typing import Callable

from torch.nn import CrossEntropyLoss
from torch.nn.modules.loss import _Loss

sys.path.append(str((Path(__file__) / "..").resolve()))
from learning_zoo import CRITERION_ZOO


def register_criterion(name: str | None = None):
    def wrapper(callable: Callable[..., _Loss]) -> Callable[..., _Loss]:
        key = callable.__name__ if name is None else name
        if key in CRITERION_ZOO:
            raise KeyError(f"An entry is already registered under the key '{key}'.")
        CRITERION_ZOO[key] = callable
        return callable

    return wrapper


register_criterion()(CrossEntropyLoss)
# TODO make Dice Loss, Focal loss
