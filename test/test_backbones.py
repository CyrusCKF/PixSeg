import sys
from pathlib import Path
from typing import cast

import torch
from torch import Tensor
from torchvision.models import resnet

sys.path.append(str((Path(__file__) / "../..").resolve()))
from src.models.backbones.resnet import ResNetBackbone


def test_resnet():
    all_model_func = [
        resnet.resnet18,
        resnet.resnet34,
        resnet.resnet50,
        resnet.resnet101,
        resnet.resnet152,
    ]
    fake_input = torch.rand([4, 3, 224, 224])
    for model_func in all_model_func:
        bb = ResNetBackbone(model_func())
        fake_output: dict[str, Tensor] = bb(fake_input)
        channels = bb.layer_channels()
        for k, out in fake_output.items():
            assert out.size(1) == channels[k]
