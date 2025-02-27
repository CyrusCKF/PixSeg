"""Contains the mapping variables for models

This file is needed due to weird global variable behaviour in Python. Please
do not access this variable directly, as it is probably empty in this namespace.
Access via `model_registry`
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Sequence
from urllib.parse import urljoin

from torch import nn
from torchvision.transforms.v2 import Transform

from ..utils.transform import SegmentationTransform


@dataclass
class SegWeights:
    file_path: str
    labels: Sequence[str]
    description: str
    base_url: str = (
        "https://github.com/CyrusCKF/semantic-segmentation-toolkit/releases/download"
    )
    transforms: Callable[..., Transform] = SegmentationTransform

    @property
    def url(self):
        return urljoin(self.base_url, self.file_path)


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
    def resolve(cls, obj: Any) -> SegWeights | None:
        """Parse and return the underlying weights data"""
        if obj is None or isinstance(obj, SegWeights):
            return obj
        if isinstance(obj, str):
            obj = cls[obj.replace(cls.__name__ + ".", "")]
        if not isinstance(obj, cls):
            raise TypeError(
                f"Invalid obj provided; expected {cls.__name__} but"
                f" received {obj.__class__.__name__}."
            )
        return obj.value


MODEL_ZOO: dict[str, Callable[..., nn.Module]] = {}
"""Mapping of model name to the model builder"""

MODEL_WEIGHTS: dict[str, type[SegWeightsEnum]] = {}
