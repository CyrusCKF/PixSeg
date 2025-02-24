import sys
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms.v2 import Transform

sys.path.append(str((Path(__file__) / "..").resolve()))
from dataset_api import DatasetMeta
from dataset_registry import register_dataset

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
_CITYSCAPES_TRAIN_IDS = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33)
_CITYSCAPES_CATEGORY_IDS:dict[str, tuple[int, ...]] = { # this is not frozen, so hide it
    "void": (0, 1, 2, 3, 4, 5, 6),
    "flat": (7, 8, 9, 10),
    "construction": (11, 12, 13, 14, 15, 16),
    "object": (17, 18, 19, 20),
    "nature": (21, 22),
    "sky": (23,),
    "human": (24, 25),
    "vehicle": (26, 27, 28, 29, 30, 31, 32, 33),
}

# fmt: on
CITYSCAPES_LABELS = tuple(
    [CITYSCAPES_FULL_LABELS[i] for i in _CITYSCAPES_TRAIN_IDS] + ["background"]
)
CITYSCAPES_COLORS = tuple(
    [CITYSCAPES_FULL_COLORS[i] for i in _CITYSCAPES_TRAIN_IDS] + [(0, 0, 0)]
)
CITYSCAPES_CATEGORY_LABELS = tuple(
    list(_CITYSCAPES_CATEGORY_IDS.keys()) + ["background"]
)
CITYSCAPES_CATEGORY_COLORS = tuple(
    [CITYSCAPES_FULL_COLORS[ids[0]] for ids in _CITYSCAPES_CATEGORY_IDS.values()]
    + [(0, 0, 0)]
)


register_dataset(
    {"target_type": "semantic", "split": "train"},
    {"target_type": "semantic", "split": "val"},
    meta=DatasetMeta(34, 255, CITYSCAPES_FULL_LABELS, CITYSCAPES_FULL_COLORS),
    name="CityscapesFull",
)(datasets.Cityscapes)


@register_dataset(
    {"target_type": "semantic", "split": "train"},
    {"target_type": "semantic", "split": "val"},
    meta=DatasetMeta(20, 255, CITYSCAPES_LABELS, CITYSCAPES_COLORS),
    name="Cityscapes",
)
class CityscapesClass(datasets.Cityscapes):
    def __init__(self, *args, **kwargs) -> None:
        """See :class:`torchvision.datasets.Cityscapes` for arguments"""
        super().__init__(*args, **kwargs)
        self.super_transforms = self.transforms
        self.transforms: Callable | None = None  # keep item in PIL.Image

    def __getitem__(self, index) -> Any:
        image, target = super().__getitem__(index)
        assert isinstance(target, Image.Image)
        target_arr = np.array(target)
        new_target = np.ones_like(target_arr) * 19  # default as background index
        for i, id_ in enumerate(_CITYSCAPES_TRAIN_IDS):
            new_target[target_arr == id_] = i
        target = Image.fromarray(new_target)

        if self.super_transforms is not None:
            image, target = self.super_transforms(image, target)
        return image, target


@register_dataset(
    {"target_type": "semantic", "split": "train"},
    {"target_type": "semantic", "split": "val"},
    meta=DatasetMeta(9, 255, CITYSCAPES_CATEGORY_LABELS, CITYSCAPES_CATEGORY_COLORS),
)
class CityscapesCategory(datasets.Cityscapes):
    def __init__(self, *args, only_train=False, **kwargs) -> None:
        """See :class:`torchvision.datasets.Cityscapes` for arguments

        Args:
            only_train: only include ids that should be trained
        """
        super().__init__(*args, **kwargs)
        self.only_train = only_train
        self.super_transforms = self.transforms
        self.transforms: Callable | None = None  # keep item in PIL.Image

    def __getitem__(self, index) -> Any:
        image, target = super().__getitem__(index)
        assert isinstance(target, Image.Image)
        target_arr = np.array(target)
        new_target = np.ones_like(target_arr) * 8  # default as background index
        for cat, ids in _CITYSCAPES_CATEGORY_IDS.items():
            for id_ in ids:
                if self.only_train and id_ not in _CITYSCAPES_TRAIN_IDS:
                    continue
                cat_id = CITYSCAPES_CATEGORY_LABELS.index(cat)
                new_target[target_arr == id_] = cat_id
        target = Image.fromarray(new_target)

        if self.super_transforms is not None:
            image, target = self.super_transforms(image, target)
        return image, target
