from dataclasses import dataclass
from functools import partial
from typing import Callable, ParamSpec, Sequence, TypeVar

from torch.utils.data import Dataset


@dataclass
class DatasetEntry:
    """Contains dataset constructor and useful information

    Attributes:
        background_index: Index in mask that should be ignored
    """

    train_constructor: Callable[..., Dataset]
    val_constructor: Callable[..., Dataset]
    num_classes: int
    background_index: int | None
    labels: Sequence[str]
    colors: Sequence[tuple[int, int, int]]

    def __init__(
        self,
        train_constructor: Callable[..., Dataset],
        val_constructor: Callable[..., Dataset],
        num_classes: int,
        background_index: int | None = None,
        labels: Sequence[str] | None = None,
        colors: Sequence[tuple[int, int, int]] | None = None,
    ):
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
"""Mapping of dataset name to a tuple of (train_constructor, val_constructor, num_classes)

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

    def wrapper(cls: Callable[P, T]) -> Callable[P, T]:
        key = cls.__name__ if name is None else name
        if key in DATASET_ZOO:
            raise ValueError(f"An entry is already registered under the name '{key}'.")
        DATASET_ZOO[key] = DatasetEntry(
            partial(cls, **train_kwargs),
            partial(cls, **val_kwargs),
            num_classes,
            background_index=background_index,
            labels=labels,
            colors=colors,
        )
        return cls

    return wrapper


def _test():
    entry = DATASET_ZOO["VOC"]
    train_dataset = entry.train_constructor(root=r"dataset", year="2007")
    print(len(train_dataset))  # type: ignore


if __name__ == "__main__":
    _test()
