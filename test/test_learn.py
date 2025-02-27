import sys
from pathlib import Path

import pytest
import torch
from torch import Tensor
from torch.utils.data import Dataset

sys.path.append(str((Path(__file__) / "../..").resolve()))
from src.semantic_segmentation_toolkit.learn import *
from src.semantic_segmentation_toolkit.learn.weighting import WeightingFunc


def test_registry():
    assert len(CLASS_WEIGHTINGS) >= 0
    assert len(CRITERION_ZOO) >= 0
    assert len(LR_SCHEDULER_ZOO) >= 0
    assert len(OPTIMIZER_ZOO) >= 0


class FakeDataset(Dataset):
    def __init__(self, num_samples, num_classes, height, width):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.height = height
        self.width = width

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx >= self.num_samples:
            raise StopIteration()
        torch.manual_seed(idx)
        image = torch.randn([3, self.height, self.width])
        mask = torch.randint(0, self.num_classes, [self.height, self.width])
        return image, mask


@pytest.fixture
def fake_dataset(num_classes=7):
    return FakeDataset(10, num_classes, 160, 90)


@pytest.mark.parametrize("weighting", CLASS_WEIGHTINGS.values())
def test_weighting(weighting: WeightingFunc, fake_dataset: FakeDataset):
    output = weighting(fake_dataset, fake_dataset.num_classes)
    assert output is None or isinstance(output, Tensor)
