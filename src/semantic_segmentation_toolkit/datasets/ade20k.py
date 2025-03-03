from pathlib import Path
from typing import Any, Literal

from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms.v2 import Transform

from .dataset_registry import register_dataset


# TODO ade20k labels and colors
@register_dataset({"split": "training"}, {"split": "validation"}, meta=150)
class ADE20K(Dataset):
    """[ADE20K](https://ade20k.csail.mit.edu/) Dataset"""

    def __init__(
        self,
        root: Path | str,
        split: Literal["training", "validation"],
        transforms: Transform | None = None,
    ) -> None:
        self.transforms = transforms

        root_path = Path(root)
        image_folder = root_path / "images" / split
        self.image_files = list(image_folder.glob("*.jpg"))
        target_folder = root_path / "annotations" / split
        self.target_files = list(target_folder.glob("*.png"))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index) -> Any:
        image = decode_image(self.image_files[index], ImageReadMode.RGB)
        target = decode_image(self.target_files[index])
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target
