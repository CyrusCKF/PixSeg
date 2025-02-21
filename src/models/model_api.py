"""Contains the mapping variables for models

This file is needed due to weird global variable behaviour in Python. Please
do not access this variable directly, as it is probably empty in this namespace.
Access via `model_registry`
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Sequence, TypeVar

from torch import nn
from torchvision.transforms.v2 import Identity, Transform


@dataclass
class SegWeights:
    url: str
    transforms: Callable[..., Transform]
    labels: Sequence[str]
    description: str


T = TypeVar("T", bound="SegWeightsEnum")


class SegWeightsEnum(Enum):
    def __init__(self, value) -> None:
        super().__init__()
        if not isinstance(value, SegWeights):
            raise TypeError(
                f"Members of {self.__class__.__name__} must be {SegWeights.__name__}"
                f" but got {value}"
            )
        self.value: SegWeights

    @classmethod
    def resolve(cls: type[T], obj: Any) -> T | None:
        if obj is None:
            return obj
        if isinstance(obj, str):
            obj = cls[obj.replace(cls.__name__ + ".", "")]
        if not isinstance(obj, cls):
            raise TypeError(
                f"Invalid obj provided; expected {cls.__name__} but"
                f" received {obj.__class__.__name__}."
            )
        return obj


MODEL_ZOO: dict[str, Callable[..., nn.Module]] = {}
"""Mapping of model name to the model builder"""
