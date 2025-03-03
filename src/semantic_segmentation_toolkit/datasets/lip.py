from pathlib import Path
from typing import Any, Literal

from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms.v2 import Transform

from .dataset_registry import DatasetMeta, register_dataset

# fmt: off
LIP_LABELS = ("Hat", "Hair", "Glove", "Sunglasses", "UpperClothes", "Dress", 
              "Coat", "Socks", "Pants", "Jumpsuits", "Scarf", "Skirt", "Face", 
              "Left-arm", "Right-arm", "Left-leg", "Right-leg", "Left-shoe", "Right-shoe")
# fmt: on


@register_dataset(
    {"split": "train"},
    {"split": "val"},
    meta=DatasetMeta.default(19, ignore_index=255, labels=LIP_LABELS),
)
class LIP(Dataset):
    """[Look into person](https://sysu-hcp.net/lip/) Dataset

    Extract every sub zip in the same folder. Final structure should look like:
    TODO LIP folder structure
    """

    def __init__(
        self,
        root: Path | str,
        split: Literal["train", "val"],
        transforms: Transform | None = None,
    ) -> None:
        self.transforms = transforms

        root_path = Path(root)
        image_folder = root_path / rf"TrainVal_images\TrainVal_images\{split}_images"
        self.image_files = list(image_folder.glob("*.jpg"))
        target_folder = (
            root_path
            / r"TrainVal_parsing_annotations\TrainVal_parsing_annotations"
            / rf"TrainVal_parsing_annotations\{split}_segmentations"
        )
        self.target_files = list(target_folder.glob("*.png"))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index) -> Any:
        image = decode_image(self.image_files[index], ImageReadMode.RGB)
        target = decode_image(self.target_files[index])
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target
