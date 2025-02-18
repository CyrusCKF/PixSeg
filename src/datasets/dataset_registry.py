import sys
from functools import partial
from pathlib import Path
from typing import Callable, ParamSpec, Sequence, TypeVar

from torch.utils.data import Dataset

sys.path.append(str((Path(__file__) / "..").resolve()))
from dataset_zoo import DATASET_ZOO, DatasetEntry

T = TypeVar("T", bound=Dataset)
P = ParamSpec("P")


def register_dataset(
    train_kwargs: dict,
    val_kwargs: dict,
    num_classes: int,
    name: str | None = None,
    ignore_index: int = -100,
    labels: Sequence[str] | None = None,
    colors: Sequence[tuple[int, int, int]] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Can be used on functions or classes"""

    def wrapper(callable: Callable[P, T]) -> Callable[P, T]:
        key = callable.__name__ if name is None else name
        if key in DATASET_ZOO:
            raise ValueError(f"An entry is already registered under the name '{key}'.")
        DATASET_ZOO[key] = DatasetEntry(
            callable,
            partial(callable, **train_kwargs),
            partial(callable, **val_kwargs),
            num_classes,
            ignore_index=ignore_index,
            labels=labels,
            colors=colors,
        )
        return callable

    return wrapper


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str((Path(__file__) / "..").resolve()))
    import pytorch_datasets

    print(DATASET_ZOO)
