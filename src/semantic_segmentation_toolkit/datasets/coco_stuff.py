from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms.v2 import Transform

from .dataset_registry import DatasetMeta, register_dataset

# fmt: off
COCO_FULL_LABELS = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", 
    "train", "truck", "boat", "traffic light", "fire hydrant", "street sign", 
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", 
    "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", 
    "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", 
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", 
    "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass", 
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", 
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", 
    "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", 
    "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", 
    "microwave", "oven", "toaster", "sink", "refrigerator", "blender", "book", 
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", 
    "hair brush", "banner", "blanket", "branch", "bridge", "building-other", 
    "bush", "cabinet", "cage", "cardboard", "carpet", "ceiling-other", "ceiling-tile", 
    "cloth", "clothes", "clouds", "counter", "cupboard", "curtain", "desk-stuff", 
    "dirt", "door-stuff", "fence", "floor-marble", "floor-other", "floor-stone", 
    "floor-tile", "floor-wood", "flower", "fog", "food-other", "fruit", 
    "furniture-other", "grass", "gravel", "ground-other", "hill", "house", "leaves", 
    "light", "mat", "metal", "mirror-stuff", "moss", "mountain", "mud", "napkin", 
    "net", "paper", "pavement", "pillow", "plant-other", "plastic", "platform", 
    "playingfield", "railing", "railroad", "river", "road", "rock", "roof", "rug", 
    "salad", "sand", "sea", "shelf", "sky-other", "skyscraper", "snow", "solid-other", 
    "stairs", "stone", "straw", "structural-other", "table", "tent", "textile-other", 
    "towel", "tree", "vegetable", "wall-brick", "wall-concrete", "wall-other", 
    "wall-panel", "wall-stone", "wall-tile", "wall-wood", "water-other", "waterdrops", 
    "window-blind", "window-other", "wood"
)
"""Index of COCO_FULL_LABELS"""
# fmt: on
_COCO_REMOVED_IDS = (11, 25, 28, 29, 44, 65, 67, 68, 70, 82, 90)
_kept_ids = tuple(
    [i for i in range(len(COCO_FULL_LABELS)) if i not in _COCO_REMOVED_IDS]
)
_VOC_IDS = (0, 1, 2, 4, 5, 6, 8, 15, 16, 17, 18, 19, 20, 43, 61, 62, 63, 66, 71)

COCO_STUFF_LABELS = tuple([COCO_FULL_LABELS[i] for i in _kept_ids] + ["unlabeled"])
COCO_VOC_LABELS = tuple([COCO_FULL_LABELS[i] for i in _VOC_IDS] + ["unlabeled"])


@register_dataset(
    {"split": "train", "include_ids": _VOC_IDS, "extra_id": 19},
    {"split": "val", "include_ids": _VOC_IDS, "extra_id": 19},
    meta=DatasetMeta.default(num_classes=20, labels=COCO_VOC_LABELS),
    name="coco{voc}",
)
@register_dataset(
    {"split": "train", "include_ids": _kept_ids, "extra_id": 171},
    {"split": "val", "include_ids": _kept_ids, "extra_id": 171},
    meta=DatasetMeta.default(num_classes=172, labels=COCO_STUFF_LABELS),
    name="coco",
)
class COCOStuff(Dataset):
    """[COCO-stuff](https://github.com/nightrome/cocostuff) Dataset

    Extract the pixel-level annotations of COCO-stuff (stuffthingmaps_trainval2017.zip)
    in the annotation folder of coco, like:

    ```
    coco
    |-  images
        |-  train2017
        |-  ...
    |-  annotations
        |-  train2017 (the labels)
        |-  val2017 (the labels)
    ```
    """

    def __init__(
        self,
        root: Path | str,
        split: Literal["train", "val", "test"],
        transforms: Transform | None = None,
        include_ids: list[int] | None = None,
        extra_id: int = 255,
    ) -> None:
        """
        Args:
            include_ids: List of ids to include in mask
            extra_id: Id to assign the remaining indices to
        """
        self.transforms = transforms
        self.include_ids = include_ids
        self.extra_id = extra_id

        root_path = Path(root)
        image_folder = root_path / f"images/{split}2017"
        self.image_files = list(image_folder.glob("*.jpg"))
        if split == "test":
            self.target_files = None
            return

        target_folder = root_path / f"annotations/{split}2017"
        self.target_files = list(target_folder.glob("*.png"))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index) -> Any:
        image = decode_image(self.image_files[index], ImageReadMode.RGB)
        if self.target_files is None:
            target = None
        else:
            full_target = decode_image(self.target_files[index])
            if self.include_ids is None:
                target = full_target
            else:
                target = torch.full_like(full_target, self.extra_id)
                for i, id_ in enumerate(self.include_ids):
                    target[full_target == id_] = i
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target
