from typing import Callable, ParamSpec, TypeVar

from torch import nn

from .model_api import MODEL_WEIGHTS, MODEL_ZOO, SegWeightsEnum

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
