from typing import cast

from torch import nn
from torchvision.models import vgg
from torchvision.models._utils import IntermediateLayerGetter


class VGGBackbone(IntermediateLayerGetter):
    def __init__(self, model: vgg.VGG) -> None:
        features = cast(nn.Sequential, model.features)
        pool_indices = [
            i for i, module in enumerate(features) if isinstance(module, nn.MaxPool2d)
        ]
        layers = [f"pool{i+1}" for i in range(len(pool_indices))]
        return_layers = {str(idx): layer for idx, layer in zip(pool_indices, layers)}
        super().__init__(features, return_layers)

    def layer_channels(self) -> dict[str, int]:
        num_channels: dict[str, int] = {}
        cur_channel = 3
        for name, module in self.named_children():
            if isinstance(module, nn.Conv2d):
                cur_channel = module.out_channels
            if name in self.return_layers:
                key = self.return_layers[name]
                num_channels[key] = cur_channel
        return num_channels
