"""Contains the `MODEL_ZOO` variable

This file is needed due to weird global variable behaviour in Python. Please
do not access this variable directly, as it is probably empty in this namespace.
Access via `model_registry`
"""

from typing import Callable

from torch import nn

MODEL_ZOO: dict[str, Callable[..., nn.Module]] = {}
"""Mapping of model name to the model builder"""
