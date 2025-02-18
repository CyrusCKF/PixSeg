from typing import Callable

from torch import nn

MODEL_ZOO: dict[str, Callable[..., nn.Module]] = {}
"""Mapping of model name to the model builder"""
