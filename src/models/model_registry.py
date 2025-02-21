import sys
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar

from torch import nn

sys.path.append(str((Path(__file__) / "..").resolve()))
from model_api import MODEL_ZOO

T = TypeVar("T", bound=nn.Module)
P = ParamSpec("P")


def register_model(name: str | None = None):
    def wrapper(func: Callable[P, T]) -> Callable[P, T]:
        key = func.__name__ if name is None else name
        if key in MODEL_ZOO:
            raise KeyError(f"An entry is already registered under the key '{key}'.")
        MODEL_ZOO[key] = func
        return func

    return wrapper


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str((Path(__file__) / "..").resolve()))
    import pytorch_models

    print(MODEL_ZOO)
