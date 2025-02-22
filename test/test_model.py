import sys
from pathlib import Path
from typing import Callable

import pytest
import torch
from torch import Tensor, nn

sys.path.append(str((Path(__file__) / "../..").resolve()))
from src.models import *


def test_registry():
    assert len(MODEL_ZOO) >= 0


@pytest.mark.parametrize("model_builder", MODEL_ZOO.values())
def test_model(model_builder: Callable[..., nn.Module]):
    model = model_builder(weights_backbone=None)
    fake_input = torch.rand([4, 3, 32, 56])
    fake_output: dict[str, Tensor] = model(fake_input)
    for k, v in fake_output.items():
        assert k in ("out", "aux")
        assert v.size(0) == fake_input.size(0)
        assert v.shape[2:] == fake_input.shape[2:]


if __name__ == "__main__":
    import torchinfo

    print(MODEL_ZOO)

    model = pspnet_resnet50()
    print(torchinfo.summary(model, [4, 3, 32, 56]))
    print(model)
