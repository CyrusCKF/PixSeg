import sys
from pathlib import Path
from typing import Callable

import pytest
import torch
from torch import Tensor, nn
from torchvision.models import *  # type: ignore

sys.path.append(str((Path(__file__) / "../..").resolve()))
from src.semantic_segmentation_toolkit.models.backbones import *

parameters = [
    (ResNetBackbone, resnet18, lambda x: x.layer_channels()),
    (ResNetBackbone, resnet34, lambda x: x.layer_channels()),
    (ResNetBackbone, resnet50, lambda x: x.layer_channels()),
    (ResNetBackbone, resnet101, lambda x: x.layer_channels()),
    (ResNetBackbone, resnet152, lambda x: x.layer_channels()),
    (MobileNetV3Backbone, mobilenet_v3_small, lambda x: x.layer_channels()),
    (MobileNetV3Backbone, mobilenet_v3_large, lambda x: x.layer_channels()),
    (VGGBackbone, vgg11, lambda x: x.layer_channels()),
    (VGGBackbone, vgg13, lambda x: x.layer_channels()),
    (VGGBackbone, vgg16, lambda x: x.layer_channels()),
    (VGGBackbone, vgg19, lambda x: x.layer_channels()),
]


@pytest.mark.parametrize("constructor, model_builder, channels_getter", parameters)
def test_backbone(
    constructor: Callable[[nn.Module], nn.Module],
    model_builder: Callable[..., nn.Module],
    channels_getter: Callable[[nn.Module], dict[str, int]],
):
    fake_input = torch.rand([4, 3, 224, 224])
    backbone = constructor(model_builder())
    fake_output: dict[str, Tensor] = backbone(fake_input)
    channels = channels_getter(backbone)
    for k, out in fake_output.items():
        assert out.size(1) == channels[k]


def _main():
    import torch
    from torchinfo import summary

    fake_input = torch.rand([4, 3, 224, 224])
    model = vgg16()
    backbone = VGGBackbone(model)
    print(backbone)

    summary(backbone, input_data=fake_input)
    fake_output = backbone(fake_input)
    for k, v in fake_output.items():
        print(k, v.dtype, v.shape)

    print(backbone.layer_channels())


if __name__ == "__main__":
    _main()
