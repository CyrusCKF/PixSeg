from typing import Callable, ParamSpec, TypeVar

from torch import nn

MODEL_ZOO: dict[str, Callable[..., nn.Module]] = {}
"""Mapping of model name to the model builder"""

T = TypeVar("T", bound=nn.Module)
P = ParamSpec("P")


def register_model(
    name: str | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def wrapper(func: Callable[P, T]) -> Callable[P, T]:
        key = func.__name__ if name is None else name
        if key in MODEL_ZOO:
            raise ValueError(f"An entry is already registered under the name '{key}'.")
        MODEL_ZOO[key] = func
        return func

    return wrapper
