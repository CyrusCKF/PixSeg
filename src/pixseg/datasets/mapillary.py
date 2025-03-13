from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms.v2 import Transform

from .dataset_registry import DatasetMeta, register_dataset

# TODO support both mapillary versions
# fmt: off
MAPILLARY_FULL_LABELS = (
    "Bird", "Ground Animal", "Ambiguous Barrier", "Concrete Block", "Curb",
    "Fence", "Guard Rail", "Barrier", "Road Median", "Road Side", "Lane Separator",
    "Temporary Barrier", "Wall", "Bike Lane", "Crosswalk - Plain", "Curb Cut",
    "Driveway", "Parking", "Parking Aisle", "Pedestrian Area", "Rail Track",
    "Road", "Road Shoulder", "Service Lane", "Sidewalk", "Traffic Island",
    "Bridge", "Building", "Garage", "Tunnel", "Person", "Person Group",
    "Bicyclist", "Motorcyclist", "Other Rider", "Lane Marking - Dashed Line",
    "Lane Marking - Straight Line", "Lane Marking - Zigzag Line",
    "Lane Marking - Ambiguous", "Lane Marking - Arrow (Left)",
    "Lane Marking - Arrow (Other)", "Lane Marking - Arrow (Right)",
    "Lane Marking - Arrow (Split Left or Straight)",
    "Lane Marking - Arrow (Split Right or Straight)",
    "Lane Marking - Arrow (Straight)", "Lane Marking - Crosswalk",
    "Lane Marking - Give Way (Row)", "Lane Marking - Give Way (Single)",
    "Lane Marking - Hatched (Chevron)", "Lane Marking - Hatched (Diagonal)",
    "Lane Marking - Other", "Lane Marking - Stop Line",
    "Lane Marking - Symbol (Bicycle)", "Lane Marking - Symbol (Other)",
    "Lane Marking - Text", "Lane Marking (only) - Dashed Line",
    "Lane Marking (only) - Crosswalk", "Lane Marking (only) - Other",
    "Lane Marking (only) - Test", "Mountain", "Sand", "Sky", "Snow", "Terrain",
    "Vegetation", "Water", "Banner", "Bench", "Bike Rack", "Catch Basin",
    "CCTV Camera", "Fire Hydrant", "Junction Box", "Mailbox", "Manhole",
    "Parking Meter", "Phone Booth", "Pothole", "Signage - Advertisement",
    "Signage - Ambiguous", "Signage - Back", "Signage - Information",
    "Signage - Other", "Signage - Store", "Street Light", "Pole", "Pole Group",
    "Traffic Sign Frame", "Utility Pole", "Traffic Cone",
    "Traffic Light - General (Single)", "Traffic Light - Pedestrians",
    "Traffic Light - General (Upright)", "Traffic Light - General (Horizontal)",
    "Traffic Light - Cyclists", "Traffic Light - Other",
    "Traffic Sign - Ambiguous", "Traffic Sign (Back)",
    "Traffic Sign - Direction (Back)", "Traffic Sign - Direction (Front)",
    "Traffic Sign (Front)", "Traffic Sign - Parking",
    "Traffic Sign - Temporary (Back)", "Traffic Sign - Temporary (Front)",
    "Trash Can", "Bicycle", "Boat", "Bus", "Car", "Caravan", "Motorcycle",
    "On Rails", "Other Vehicle", "Trailer", "Truck", "Vehicle Group",
    "Wheeled Slow", "Water Valve", "Car Mount", "Dynamic", "Ego Vehicle", "Ground",
    "Static", "Unlabeled"
)
MAPILLARY_FULL_COLORS = (
    (165, 42, 42), (0, 192, 0), (250, 170, 31), (250, 170, 32), (196, 196, 196),
    (190, 153, 153), (180, 165, 180), (90, 120, 150), (250, 170, 33),
    (250, 170, 34), (128, 128, 128), (250, 170, 35), (102, 102, 156),
    (128, 64, 255), (140, 140, 200), (170, 170, 170), (250, 170, 36),
    (250, 170, 160), (250, 170, 37), (96, 96, 96), (230, 150, 140), (128, 64, 128),
    (110, 110, 110), (110, 110, 110), (244, 35, 232), (128, 196, 128),
    (150, 100, 100), (70, 70, 70), (150, 150, 150), (150, 120, 90), (220, 20, 60),
    (220, 20, 60), (255, 0, 0), (255, 0, 100), (255, 0, 200), (255, 255, 255),
    (255, 255, 255), (250, 170, 29), (250, 170, 28), (250, 170, 26),
    (250, 170, 25), (250, 170, 24), (250, 170, 22), (250, 170, 21), (250, 170, 20),
    (255, 255, 255), (250, 170, 19), (250, 170, 18), (250, 170, 12),
    (250, 170, 11), (255, 255, 255), (255, 255, 255), (250, 170, 16),
    (250, 170, 15), (250, 170, 15), (255, 255, 255), (255, 255, 255),
    (255, 255, 255), (255, 255, 255), (64, 170, 64), (230, 160, 50),
    (70, 130, 180), (190, 255, 255), (152, 251, 152), (107, 142, 35), (0, 170, 30),
    (255, 255, 128), (250, 0, 30), (100, 140, 180), (220, 128, 128), (222, 40, 40),
    (100, 170, 30), (40, 40, 40), (33, 33, 33), (100, 128, 160), (20, 20, 255),
    (142, 0, 0), (70, 100, 150), (250, 171, 30), (250, 172, 30), (250, 173, 30),
    (250, 174, 30), (250, 175, 30), (250, 176, 30), (210, 170, 100),
    (153, 153, 153), (153, 153, 153), (128, 128, 128), (0, 0, 80), (210, 60, 60),
    (250, 170, 30), (250, 170, 30), (250, 170, 30), (250, 170, 30), (250, 170, 30),
    (250, 170, 30), (192, 192, 192), (192, 192, 192), (192, 192, 192),
    (220, 220, 0), (220, 220, 0), (0, 0, 196), (192, 192, 192), (220, 220, 0),
    (140, 140, 20), (119, 11, 32), (150, 0, 255), (0, 60, 100), (0, 0, 142),
    (0, 0, 90), (0, 0, 230), (0, 80, 100), (128, 64, 64), (0, 0, 110), (0, 0, 70),
    (0, 0, 142), (0, 0, 192), (170, 170, 170), (32, 32, 32), (111, 74, 0),
    (120, 10, 10), (81, 0, 81), (111, 111, 0), (0, 0, 0)
)
# fmt: on
_MAPILLARY_NOT_EVALUATED = (2, 31, 38, 79, 86, 96, 115, 123)

_train_ids = tuple(
    [i for i in range(len(MAPILLARY_FULL_LABELS)) if i not in _MAPILLARY_NOT_EVALUATED]
)
MAPILLARY_LABELS = tuple([MAPILLARY_FULL_LABELS[i] for i in _train_ids])
MAPILLARY_COLORS = tuple([MAPILLARY_FULL_COLORS[i] for i in _train_ids])


@register_dataset(
    {"split": "training"},
    {"split": "validation"},
    meta=DatasetMeta(116, 255, MAPILLARY_LABELS, MAPILLARY_COLORS),
    name="Mapillary",
)
class MapillaryVistas(Dataset):
    """[Mapillary Vistas](https://www.mapillary.com/dataset/vistas) Dataset"""

    def __init__(
        self,
        root: Path | str,
        split: Literal["training", "validation", "testing"],
        transforms: Transform | None = None,
        version: str = "v2.0",
    ) -> None:
        self.transforms = transforms

        root_path = Path(root)
        image_folder = root_path / rf"{split}\images"
        self.image_files = list(image_folder.glob("*.jpg"))
        if split == "testing":
            self.target_files = None
            return

        target_folder = root_path / rf"{split}\{version}\labels"
        self.target_files = list(target_folder.glob("*.png"))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index) -> Any:
        image = decode_image(self.image_files[index], ImageReadMode.RGB)
        if self.target_files is None:
            target = None
        else:
            full_target = decode_image(self.target_files[index])
            target = torch.full_like(full_target, 255)
            for i, id_ in enumerate(_train_ids):
                target[full_target == id_] = i

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target
