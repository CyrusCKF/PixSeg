import sys
import warnings
from pathlib import Path

import pytest
import toml
import torch
from torch import Tensor
from torch.utils.data import Subset

sys.path.append(str((Path(__file__) / "../..").resolve()))
from src.datasets import DATASET_METADATA, DATASET_ZOO, resolve_metadata
from src.utils.transform import DataTransform


def test_registry():
    assert len(DATASET_ZOO) >= 0
    assert len(DATASET_METADATA) >= 0
    assert DATASET_ZOO.keys() == DATASET_METADATA.keys()


@pytest.fixture
def dataset_roots(file="test/dataset_root.toml") -> dict[str, str]:
    root_file = Path(file)
    if not root_file.is_file():
        raise RuntimeError(
            f"Dataset root file {root_file} not found."
            f" Please copy from doc/pytest_dataset_root.toml"
        )
    return toml.load(root_file)


@pytest.mark.parametrize("name", DATASET_ZOO.keys())
def test_dataset_with_format(name, dataset_roots, test_size=10):
    if name not in dataset_roots or dataset_roots[name] == "":
        warnings.warn(f"Root of dataset {name} is not specified. Skipping the test")
        return

    entry = DATASET_ZOO[name]
    meta = resolve_metadata(name)
    root = dataset_roots[name]
    transform = DataTransform()
    datasets = [
        entry.construct_train(root, transform),
        entry.construct_val(root, transform),
    ]

    for dataset in datasets:
        dataset_size: int = len(dataset)  # type: ignore
        assert dataset_size > 0
        # only test a subset
        subset = Subset(dataset, range(min(dataset_size, test_size)))
        for image, mask in iter(subset):
            assert isinstance(image, Tensor) and image.dtype == torch.float32
            C, H, W = image.shape
            assert C == 3, "Images should have 3 channels"

            assert (
                isinstance(mask, Tensor)
                and mask.dtype == torch.long
                and mask.shape == (H, W)
            )
            is_class = (mask >= 0) & (mask < meta.num_classes)
            is_ignored = mask == meta.ignore_index
            unexpected = torch.masked_select(mask, ~(is_class | is_ignored))
            unique_unexpected = torch.unique(unexpected).tolist()
            assert (
                unexpected.numel() == 0
            ), f"Unexpected classes in dataset {name}: {unique_unexpected}"


if __name__ == "__main__":
    print(DATASET_ZOO)
    print(DATASET_METADATA)
