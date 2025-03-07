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


@pytest.fixture
def fake_inputs():
    return [
        torch.rand([2, 3, 64, 64]),
        torch.rand([2, 3, 61, 63]),
        torch.rand([2, 3, 65, 67]),
    ]


@pytest.mark.parametrize("model_builder", MODEL_ZOO.values())
def test_model(fake_inputs, model_builder: Callable[..., nn.Module]):
    # disable backbone weights if needed
    try:
        model = model_builder(weights_backbone=None)
    except TypeError:
        model = model_builder()
    for fake_input in fake_inputs:
        fake_output: dict[str, Tensor] = model(fake_input)
        for k, v in fake_output.items():
            assert k in ("out", "aux")
            assert v.size(0) == fake_input.size(0)
            assert v.shape[2:] == fake_input.shape[2:]


def _main():
    from pprint import pprint

    import torchinfo

    from src.semantic_segmentation_toolkit.datasets import CITYSCAPES_LABELS, VOC_LABELS

    # model = pspnet_resnet50(num_classes=len(CITYSCAPES_LABELS))
    # model.eval()
    # print(model)
    # torchinfo.summary(model, [1, 3, 512, 1024])
    # pprint(MODEL_ZOO.keys(), compact=True)
    # for key, weights in MODEL_WEIGHTS.items():
    #     print(key, [w.name for w in weights])

    nets = [upernet_resnet18(num_classes=10), upernet_resnet101(num_classes=10)]
    for net in nets:
        fake_input = torch.rand([4, 3, 512, 512])
        output = net(fake_input)
        print("output", output["out"].shape)
        torchinfo.summary(
            net,
            input_data=fake_input,
            col_names=("output_size", "num_params", "mult_adds"),
        )


if __name__ == "__main__":
    from src.semantic_segmentation_toolkit.models.upernet import *

    _main()
