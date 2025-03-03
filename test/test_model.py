import sys
from pathlib import Path
from typing import Callable

import pytest
import torch
from torch import Tensor, nn

sys.path.append(str((Path(__file__) / "../..").resolve()))
from src.semantic_segmentation_toolkit.models import *


def test_registry():
    assert len(MODEL_ZOO) >= 0
    assert len(MODEL_WEIGHTS) >= 0
    assert set(MODEL_ZOO.keys()).issuperset(set(MODEL_WEIGHTS.keys()))


@pytest.mark.parametrize("model_builder", MODEL_ZOO.values())
def test_model(model_builder: Callable[..., nn.Module]):
    # disable backbone weights if needed
    try:
        model = model_builder(weights_backbone=None)
    except TypeError:
        model = model_builder()
    fake_input = torch.rand([4, 3, 32, 56])
    fake_output: dict[str, Tensor] = model(fake_input)
    for k, v in fake_output.items():
        assert k in ("out", "aux")
        assert v.size(0) == fake_input.size(0)
        assert v.shape[2:] == fake_input.shape[2:]


def _main():
    from pprint import pprint

    import torchinfo

    model = enet_original()
    torchinfo.summary(model, [4, 3, 31, 55])
    print(model)

    pprint(MODEL_ZOO)
    for key, weights in MODEL_WEIGHTS.items():
        print(key, [(w.name, w.value.url) for w in weights])


if __name__ == "__main__":
    _main()
