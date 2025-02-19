import sys
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar

from torch import optim
from torch.optim import Optimizer

sys.path.append(str((Path(__file__) / "..").resolve()))
from learning_zoo import OPTIMIZER_ZOO

T = TypeVar("T", bound=Optimizer)
P = ParamSpec("P")


def register_optimizer(callable: Callable[P, T]) -> Callable[P, T]:
    key = callable.__name__
    if key in OPTIMIZER_ZOO:
        raise ValueError(f"An entry is already registered under the name '{key}'.")
    OPTIMIZER_ZOO[key] = callable
    return callable


register_optimizer(optim.Adam)
register_optimizer(optim.SGD)
# TODO implement padam
