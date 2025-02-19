import sys
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar

from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LRScheduler

sys.path.append(str((Path(__file__) / "..").resolve()))
from learning_zoo import LR_SCHEDULER_ZOO

T = TypeVar("T", bound=LRScheduler)
P = ParamSpec("P")


def register_lr_scheduler(callable: Callable[P, T]) -> Callable[P, T]:
    key = callable.__name__
    if key in LR_SCHEDULER_ZOO:
        raise ValueError(f"An entry is already registered under the name '{key}'.")
    LR_SCHEDULER_ZOO[key] = callable
    return callable


register_lr_scheduler(lr_scheduler.StepLR)
register_lr_scheduler(lr_scheduler.PolynomialLR)
register_lr_scheduler(lr_scheduler.OneCycleLR)
register_lr_scheduler(lr_scheduler.CosineAnnealingLR)
