import sys
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

sys.path.append(str((Path(__file__) / "../..").resolve()))
from src.models.backbones import resnet


def test_resnet():
    all_func = [resnet.resnet34, resnet.resnet50, resnet.resnet101, resnet.resnet152]
    fake_input = torch.rand([4, 3, 224, 224])
    for func in all_func:
        bb = cast(resnet.ResNetBackbone, func(weights=None))
        fake_output: dict[str, Tensor] = bb(fake_input)
        channels = bb.layer_channels()
        for k, out in fake_output.items():
            assert out.size(1) == channels[k]
