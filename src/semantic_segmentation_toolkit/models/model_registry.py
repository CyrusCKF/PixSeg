import typing
from dataclasses import dataclass
from enum import Enum
from inspect import signature
from typing import Any, Callable, ParamSpec, Sequence, TypeVar

from torch import nn
from torchvision.transforms.v2 import Transform

from ..utils.transform import SegmentationTransform


@dataclass
class SegWeights:
    file_path: str
    labels: Sequence[str]
    description: str
    base_url: str = (
        "https://github.com/CyrusCKF/semantic-segmentation-toolkit/releases/download/"
    )
    transforms: Callable[..., Transform] = SegmentationTransform

    @property
    def url(self):
        return self.base_url + self.file_path


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
            # use "try/except" not "in" in case of duplicate enum member
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self._name_}"


MODEL_ZOO: dict[str, Callable[..., nn.Module]] = {}
"""Mapping of model name to the model builder"""

MODEL_WEIGHTS: dict[str, type[SegWeightsEnum]] = {}

T = TypeVar("T", bound=nn.Module)
P = ParamSpec("P")


def register_model(name: str | None = None):
    def wrapper(func: Callable[P, T]) -> Callable[P, T]:
        key = func.__name__ if name is None else name
        if key in MODEL_ZOO:
            raise KeyError(f"An entry is already registered under the key '{key}'.")
        MODEL_ZOO[key] = func

        # try to infer weights enum and register it
        weights_enum = _infer_weights_enum_from_builder(func)
        if weights_enum is not None:
            MODEL_WEIGHTS[key] = weights_enum
        return func

    return wrapper


def _infer_weights_enum_from_builder(builder: Callable) -> type[SegWeightsEnum] | None:
    """Try to infer weights enum from function signature"""
    sig = signature(builder)
    if "weights" not in sig.parameters:
        return None
    param = sig.parameters["weights"]
    types = typing.get_args(param.annotation) + (param.annotation,)
    for t in types:
        if isinstance(t, type) and issubclass(t, SegWeightsEnum):
            return t
    return None


"""Following functions are implemented to support 
APIs similar to https://pytorch.org/vision/main/models.html"""


def get_weight(model: str, name: str) -> SegWeights:
    """Get the model weights by its name. Example: get_weight("")"""
    weights = get_model_weights(model)[name].value
    assert isinstance(weights, SegWeights)
    return weights


def get_model_weights(model: str) -> type[SegWeightsEnum]:
    """Return the weights enum class associated to the given model."""
    if model not in MODEL_WEIGHTS:
        if model not in MODEL_ZOO:
            raise ValueError(
                f"Unknown model name {model}. Here are the available models: {MODEL_ZOO.keys()}"
            )
        else:
            raise ValueError(f"Model {model} does not have pretrained weights.")
    return MODEL_WEIGHTS[model]


def list_models():
    """Return a list with the names of registered models."""
    return list(MODEL_ZOO.keys())


def get_model_builder(name: str) -> Callable[..., nn.Module]:
    """Gets the model name and returns the model builder method."""
    if name not in MODEL_ZOO:
        raise ValueError(
            f"Unknown model name {name}. Here are the available models: {MODEL_ZOO.keys()}"
        )
    return MODEL_ZOO[name]


def get_model(name: str, **config) -> nn.Module:
    """Get the model name and configuration and return an instantiated model."""
    return get_model_builder(name)(**config)
