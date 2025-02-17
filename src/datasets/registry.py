from dataclasses import dataclass
from functools import partial
from typing import Callable, ParamSpec, Sequence, TypeVar

from torch.utils.data import Dataset


@dataclass
class DatasetEntry:
    """Contains dataset constructor and useful information

    Attributes:
        base_constructor: The original constructor of the registered dataset
        train_constructor: Constructor to create training dataset
        val_constructor: Constructor to create validation dataset
        background_index: Index in mask that should be ignored
    """

    base_constructor: Callable[..., Dataset]
    train_constructor: Callable[..., Dataset]
    val_constructor: Callable[..., Dataset]
    num_classes: int
    background_index: int | None
    labels: Sequence[str]
    colors: Sequence[tuple[int, int, int]]

    def __init__(
        self,
        base_constructor: Callable[..., Dataset],
        train_constructor: Callable[..., Dataset],
        val_constructor: Callable[..., Dataset],
        num_classes: int,
        background_index: int | None = None,
        labels: Sequence[str] | None = None,
        colors: Sequence[tuple[int, int, int]] | None = None,
    ):
        self.base_constructor = base_constructor
        self.train_constructor = train_constructor
        self.val_constructor = val_constructor
        self.num_classes = num_classes
        self.background_index = background_index
        self.labels = (
            [f"Class {i}" for i in range(self.num_classes)]
            if labels is None
            else labels
        )
        self.colors = [] if colors is None else colors  # TODO


DATASET_ZOO: dict[str, DatasetEntry] = {}
"""Mapping of dataset name to `DatabaseEntry`

All datasets must have the kwargs root (Path) and transforms (Callable)
"""

T = TypeVar("T", bound=Dataset)
P = ParamSpec("P")


def register_dataset(
    train_kwargs: dict,
    val_kwargs: dict,
    num_classes: int,
    name: str | None = None,
    background_index: int | None = None,
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
            background_index=background_index,
            labels=labels,
            colors=colors,
        )
        return callable

    return wrapper
