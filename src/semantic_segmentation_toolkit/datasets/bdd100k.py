import sys
from pathlib import Path
from typing import Any, Literal

from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms.v2 import Transform

sys.path.append(str((Path(__file__) / "..").resolve()))
from dataset_registry import register_dataset


@register_dataset({"split": "train"}, {"split": "val"}, meta="Cityscapes")
class BDD100K(Dataset):
    """[BDD100K](https://github.com/bdd100k/bdd100k) Dataset"""

    def __init__(
        self,
        root: Path | str,
        split: Literal["train", "val"],
        transforms: Transform | None = None,
    ) -> None:
        self.transforms = transforms

        root_path = Path(root)
        image_folder = root_path / rf"images\10k\{split}"
        image_files = list(image_folder.glob("*.jpg"))
        target_folder = root_path / rf"labels\sem_seg\masks\{split}"
        target_files = list(target_folder.glob("*.png"))

        # some files may be incorrect
        image_stems = [p.stem for p in image_files]
        target_stems = [p.stem for p in target_files]
        self.image_files = [p for p in image_files if p.stem in target_stems]
        self.target_files = [p for p in target_files if p.stem in image_stems]
        if len(self.image_files) != len(self.target_files):
            raise ValueError(
                f"Mismatch number of files, got {len(self.image_files)}"
                f" images and {len(self.target_files)} colormaps"
            )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index) -> Any:
        image = decode_image(self.image_files[index], ImageReadMode.RGB)
        target = decode_image(self.target_files[index])
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target
