from typing import Callable

from torch import Tensor, nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset

WeightingFunc = Callable[[Dataset, int], Tensor | None]
CLASS_WEIGHTINGS: dict[str, WeightingFunc] = {}
"""Each class weighting strategy takes in a Dataset and number of classes 
and returns a list of weighting or None"""

CRITERION_ZOO: dict[str, Callable[..., _Loss]] = {}
"""Criterions must accept kwargs `weight (Tensor)` and `ignore_index (int)`"""

OPTIMIZER_ZOO: dict[str, Callable[..., Optimizer]] = {}
LR_SCHEDULER_ZOO: dict[str, Callable[..., LRScheduler]] = {}
