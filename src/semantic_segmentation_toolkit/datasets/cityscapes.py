from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision.transforms.v2 import Transform

from .dataset_registry import DatasetMeta, register_dataset

# fmt: off
CITYSCAPES_FULL_LABELS = (
    "unlabeled", "ego vehicle", "rectification border", "out of roi", "static", 
    "dynamic", "ground", "road", "sidewalk", "parking", "rail track", "building", 
    "wall", "fence", "guard rail", "bridge", "tunnel", "pole", "polegroup", 
    "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", 
    "rider", "car", "truck", "bus", "caravan", "trailer", "train", "motorcycle", 
    "bicycle"
)
CITYSCAPES_FULL_COLORS = (
    (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (111, 74, 0), 
    (81, 0, 81), (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), 
    (70, 70, 70), (102, 102, 156), (190, 153, 153), (180, 165, 180), (150, 100, 100), 
    (150, 120, 90), (153, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0), 
    (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), 
    (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), 
    (0, 0, 230), (119, 11, 32)
)
_TRAIN_IDS = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33)
_CATEGORY_IDS:dict[str, tuple[int, ...]] = { # this is not frozen, so hide it
    "flat": (7, 8, 9, 10),
    "construction": (11, 12, 13, 14, 15, 16),
    "object": (17, 18, 19, 20),
    "nature": (21, 22),
    "sky": (23,),
    "human": (24, 25),
    "vehicle": (26, 27, 28, 29, 30, 31, 32, 33),
}
# fmt: on
# Inlcude background classes
_groups = tuple([(i,) for i in _TRAIN_IDS])
CITYSCAPES_LABELS = tuple(
    [CITYSCAPES_FULL_LABELS[i] for i in _TRAIN_IDS] + ["background"]
)
CITYSCAPES_COLORS = tuple([CITYSCAPES_FULL_COLORS[i] for i in _TRAIN_IDS] + [(0, 0, 0)])

_cat_groups = tuple(_CATEGORY_IDS.values())
CITYSCAPES_CATEGORY_LABELS = tuple(_CATEGORY_IDS.keys()) + ("background",)


register_dataset(
    {"target_type": "semantic", "split": "train"},
    {"target_type": "semantic", "split": "val"},
    meta=DatasetMeta(34, 255, CITYSCAPES_FULL_LABELS, CITYSCAPES_FULL_COLORS),
    name="Cityscapes{full}",
)(datasets.Cityscapes)


@register_dataset(
    {"split": "train", "class_groups": _groups, "extra_index": 19},
    {"split": "val", "class_groups": _groups, "extra_index": 19},
    meta=DatasetMeta(20, 255, CITYSCAPES_LABELS, CITYSCAPES_COLORS),
    name="Cityscapes",
)
@register_dataset(
    {"split": "train", "class_groups": _cat_groups, "extra_index": 7},
    {"split": "val", "class_groups": _cat_groups, "extra_index": 7},
    meta=DatasetMeta.default(8, labels=CITYSCAPES_CATEGORY_LABELS),
    name="Cityscapes{category}",
)
class CityscapesSubset(datasets.Cityscapes):
    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "test"],
        class_groups: Sequence[Sequence[int]],
        extra_id: int,
        mode: Literal["fine", "coarse"] = "fine",
        transforms: Transform | None = None,
    ) -> None:
        """Cityscapes dataset that only includes a subset of classes

        Args:
            class_groups: Group of classes to use. Each group will be mapped to the same id
            extra_id: ID for the remaining classes
            **kwargs: See :class:`torchvision.datasets.Cityscapes` for other arguments
        """
        super().__init__(root, split, mode, target_type="semantic")
        self.final_transforms = transforms  # keep parent get item the same type
        self.class_groups = class_groups
        self.extra_id = extra_id

    def __getitem__(self, index) -> Any:
        image, target = super().__getitem__(index)
        assert isinstance(target, Image.Image)
        target_arr = np.array(target)
        new_target = np.full_like(target_arr, self.extra_id)
        for new_id, classes in enumerate(self.class_groups):
            for old_id in classes:
                new_target[target_arr == old_id] = new_id
        target = Image.fromarray(new_target)

        if self.final_transforms is not None:
            image, target = self.final_transforms(image, target)
        return image, target
