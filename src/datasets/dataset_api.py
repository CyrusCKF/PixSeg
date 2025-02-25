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
        final_colors = colors or _generate_voc_palette(num_classes)
        return DatasetMeta(num_classes, ignore_index, final_labels, final_colors)


@dataclass
class DatasetEntry:
    """Contain dataset constructors"""

    constructor: Callable[..., Dataset]
    train_kwargs: dict
    val_kwargs: dict

    def construct_train(self, transforms: Callable | None = None, *args, **kwargs):
        return self.constructor(
            transforms=transforms, *args, **kwargs, **self.train_kwargs
        )

    def construct_val(self, transforms: Callable | None = None, *args, **kwargs):
        return self.constructor(
            transforms=transforms, *args, **kwargs, **self.val_kwargs
        )


DATASET_METADATA: dict[str, DatasetMeta | str] = {}
"""Mapping of meta key to the metadata or key of other metadata"""

DATASET_ZOO: dict[str, DatasetEntry] = {}
"""Mapping of dataset name to `DatabaseEntry`

All dataset constructors must accept kwargs `root (Path|str)` and `transforms (Callable|None)`
"""


def _generate_voc_palette(num_classes) -> Sequence[tuple[int, int, int]]:
    """Reference https://github.com/yassouali/pytorch-segmentation/blob/master/utils/palette.py"""
    palette: list[tuple[int, int, int]] = []
    for j in range(0, num_classes):
        color = [0, 0, 0]
        lab = j
        i = 0
        while lab > 0:
            color[0] |= ((lab >> 0) & 1) << (7 - i)
            color[1] |= ((lab >> 1) & 1) << (7 - i)
            color[2] |= ((lab >> 2) & 1) << (7 - i)
            i = i + 1
            lab >>= 3
        color_tuple = (color[0], color[1], color[2])
        palette.append(color_tuple)
    return palette
