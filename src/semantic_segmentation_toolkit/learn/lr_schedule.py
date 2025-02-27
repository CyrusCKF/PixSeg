from typing import Callable, ParamSpec, TypeVar

from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LRScheduler

from .learn_api import LR_SCHEDULER_ZOO

T = TypeVar("T", bound=LRScheduler)
P = ParamSpec("P")


def register_lr_scheduler(name: str | None = None):
    def wrapper(callable: Callable[P, T]) -> Callable[P, T]:
        key = callable.__name__ if name is None else name
        if key in LR_SCHEDULER_ZOO:
            raise KeyError(f"An entry is already registered under the key '{key}'.")
        LR_SCHEDULER_ZOO[key] = callable
        return callable

    return wrapper


register_lr_scheduler()(lr_scheduler.StepLR)
register_lr_scheduler()(lr_scheduler.PolynomialLR)
register_lr_scheduler()(lr_scheduler.OneCycleLR)
register_lr_scheduler()(lr_scheduler.CosineAnnealingLR)
