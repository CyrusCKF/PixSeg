from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from torch.utils.data import Dataset


@dataclass
class DatasetMeta:
    """Contain dataset useful information

    Attributes:
        ignore_index: Index in mask that should be ignored
    """

    num_classes: int
    ignore_index: int
    labels: Sequence[str]
    colors: Sequence[tuple[int, int, int]]

    def __post_init__(self):
        if self.num_classes != len(self.labels):
            raise ValueError(
                f"Mismatch size of labels, expected {self.num_classes}, but got {len(self.labels)}"
            )
        if self.num_classes != len(self.colors):
            raise ValueError(
                f"Mismatch size of colors, expected {self.num_classes}, but got {len(self.colors)}"
            )

    @staticmethod
    def default(
        num_classes: int,
        ignore_index=-100,
        labels: Sequence[str] | None = None,
        colors: Sequence[tuple[int, int, int]] | None = None,
    ) -> "DatasetMeta":
        final_labels = labels or [f"Class {i}" for i in range(num_classes)]
        # TODO better default colors
        final_colors = colors or [(255, 255, 255) for i in range(num_classes)]
        return DatasetMeta(num_classes, ignore_index, final_labels, final_colors)


@dataclass
class DatasetEntry:
    """Contain dataset constructors"""

    constructor: Callable[..., Dataset]
    train_kwargs: dict
    val_kwargs: dict

    def construct_train(
        self, root: Path | str, transforms: Callable | None = None, *args, **kwargs
    ):
        return self.constructor(
            root=root, transforms=transforms, *args, **kwargs, **self.train_kwargs
        )

    def construct_val(
        self, root: Path | str, transforms: Callable | None = None, *args, **kwargs
    ):
        return self.constructor(
            root=root, transforms=transforms, *args, **kwargs, **self.val_kwargs
        )


DATASET_METADATA: dict[str, DatasetMeta | str] = {}
"""Mapping of meta key to the metadata or key of other metadata"""

DATASET_ZOO: dict[str, DatasetEntry] = {}
"""Mapping of dataset name to `DatabaseEntry`

All dataset constructors must accept kwargs `root (Path|str)` and `transforms (Callable|None)`
"""
