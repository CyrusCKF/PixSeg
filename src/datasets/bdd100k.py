import sys
from pathlib import Path
from typing import Literal

from torch.utils.data import Dataset
from torchvision.transforms.v2 import Transform

sys.path.append(str((Path(__file__) / "..").resolve()))
from dataset_registry import register_dataset


# TODO BDD100K
@register_dataset({"split": "train"}, {"split": "val"}, meta="Cityscapes")
class BDD100K(Dataset):
    """Dataset for bdd100k"""

    def __init__(
        self,
        root: Path | str,
        split: Literal["train", "val"],
        transforms: Transform | None = None,
    ) -> None:
        self.split = split
        self.root = Path(root)
        self.transforms = transforms
        self.image_files, self.mask_files = self._populate_files()

    def _populate_files(self) -> tuple[list[Path], list[Path]]:
        image_folder = self.root / rf"images\10k\{self.split}"
        image_files = list(image_folder.glob("*.jpg"))
        mask_folder = self.root / rf"labels\sem_seg\colormaps\{self.split}"
        mask_files = list(mask_folder.glob("*.png"))

        # some files may be incorrect
        image_stems = [p.stem for p in image_files]
        mask_stems = [p.stem for p in mask_files]
        image_files = [p for p in image_files if p.stem in mask_stems]
        mask_files = [p for p in mask_files if p.stem in image_stems]

        return image_files, mask_files
