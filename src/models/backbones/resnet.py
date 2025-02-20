from typing import cast

from torch import nn
from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter


class ResNetBackbone(IntermediateLayerGetter):
    def __init__(self, model: resnet.ResNet) -> None:
        layers = [f"layer{i+1}" for i in range(4)]
        return_layers = {layer: layer for layer in layers}
        super().__init__(model, return_layers)

    def layer_channels(self) -> dict[str, int]:
        num_channels: dict[str, int] = {}
        for name, module in self.named_children():
            if name not in self.return_layers:
                continue

            key = self.return_layers[name]
            last_block = cast(nn.Sequential, module)[-1]
            if isinstance(last_block, resnet.BasicBlock):
                num_channels[key] = cast(int, last_block.bn2.num_features)
            elif isinstance(last_block, resnet.Bottleneck):
                num_channels[key] = cast(int, last_block.bn3.num_features)
            else:
                raise ValueError(f"Unknown block {key} of type {type(module)}")
        return num_channels


def resnet18(weights=resnet.ResNet18_Weights.DEFAULT, **kwargs):
    model = resnet.resnet18(weights=weights, **kwargs)
    backbone = ResNetBackbone(model)
    return backbone


def resnet34(weights=resnet.ResNet34_Weights.DEFAULT, **kwargs):
    model = resnet.resnet34(weights=weights, **kwargs)
    backbone = ResNetBackbone(model)
    return backbone


def resnet50(weights=resnet.ResNet50_Weights.DEFAULT, **kwargs):
    kwargs["replace_stride_with_dilation"] = [False, True, True]
    model = resnet.resnet50(weights=weights, **kwargs)
    backbone = ResNetBackbone(model)
    return backbone


def resnet101(weights=resnet.ResNet101_Weights.DEFAULT, **kwargs):
    kwargs["replace_stride_with_dilation"] = [False, True, True]
    model = resnet.resnet101(weights=weights, **kwargs)
    backbone = ResNetBackbone(model)
    return backbone


def resnet152(weights=resnet.ResNet152_Weights.DEFAULT, **kwargs):
    kwargs["replace_stride_with_dilation"] = [False, True, True]
    model = resnet.resnet152(weights=weights, **kwargs)
    backbone = ResNetBackbone(model)
    return backbone


def _test():
    import torch
    from torchinfo import summary

    fake_input = torch.rand([4, 3, 224, 224])
    model = resnet101()
    summary(model, input_data=fake_input)
    print(model)
    fake_output = model(fake_input)
    for k, v in fake_output.items():
        print(k, v.dtype, v.shape)

    print(model.layer_channels())


if __name__ == "__main__":
    _test()
