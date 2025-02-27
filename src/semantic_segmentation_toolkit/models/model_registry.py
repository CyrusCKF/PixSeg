from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, ParamSpec, Sequence, TypeVar
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
            weight_name = obj.replace(cls.__name__ + ".", "")
            # use try/catch not other checking because duplicate enum member cannot be found
            try:
                obj = cls[weight_name]
            except:
                raise ValueError(
                    f"Failed to find Weights {weight_name} in {cls.__name__}."
                    f" Try one of {[e.name for e in cls]}"
                ) from None
        if not isinstance(obj, cls):
            raise TypeError(
                f"Invalid obj provided; expected {cls.__name__} but"
                f" received {obj.__class__.__name__}."
            )
        return obj.value


MODEL_ZOO: dict[str, Callable[..., nn.Module]] = {}
"""Mapping of model name to the model builder"""

MODEL_WEIGHTS: dict[str, type[SegWeightsEnum]] = {}

T = TypeVar("T", bound=nn.Module)
P = ParamSpec("P")


def register_model(
    name: str | None = None, weights_enum: type[SegWeightsEnum] | None = None
):
    def wrapper(func: Callable[P, T]) -> Callable[P, T]:
        key = func.__name__ if name is None else name
        if key in MODEL_ZOO or key in MODEL_WEIGHTS:
            raise KeyError(f"An entry is already registered under the key '{key}'.")

        MODEL_ZOO[key] = func
        if weights_enum is not None:
            MODEL_WEIGHTS[key] = weights_enum
        return func

    return wrapper
