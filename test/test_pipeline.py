"""Since pipeline integrates different components, this is pretty much an integration test."""

import shutil
import sys
import warnings
from pathlib import Path

import toml
import torch
from torch.utils.data import Dataset

sys.path.append(str((Path(__file__) / "../..").resolve()))
from src.semantic_segmentation_toolkit.datasets import register_dataset
from src.semantic_segmentation_toolkit.pipeline import Config
from src.semantic_segmentation_toolkit.utils.rng import seed

NUM_FAKE_CLASSES = 10


@register_dataset({}, {}, NUM_FAKE_CLASSES)
class _FakeDataset(Dataset):
    def __init__(self, transforms, num_samples=20, height=480, width=640):
        self.num_samples = num_samples
        self.height = height
        self.width = width
        self.transforms = transforms
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if index >= self.num_samples:
            raise StopIteration()
        seed(index)
        image = torch.randn([3, self.height, self.width])
        mask = torch.randint(0, NUM_FAKE_CLASSES, [1, self.height, self.width])
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        return image, mask


def test_config_trainer(path=r"doc\sample_config.toml"):
    config_dict = toml.load(path)
    config_dict["data"]["dataset"]["dataset"] = "_FakeDataset"
    config_dict["data"]["dataset"]["params"] = {}
    config_dict["trainer"]["params"]["num_epochs"] = 1
    config = Config(config_dict)
    trainer = config.to_trainer()
    trainer.train()

    # clean up
    shutil.rmtree(config.out_folder)
    if config.out_folder.is_dir():
        warnings.warn(
            f"Test completed successfully, but outputs are not cleant up."
            f" Please check the output folder {config.out_folder}"
        )


def _main():
    import logging

    import toml

    logging.basicConfig(level=logging.DEBUG)

    config_toml = toml.load(r"doc\sample_config.toml")
    config_toml["data"]["dataset"]["params"]["root"] = r"dataset"
    config = Config(config_toml)
    trainer = config.to_trainer()


if __name__ == "__main__":
    _main()
