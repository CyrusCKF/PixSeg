from typing import cast

from torch import nn
from torchvision.models import mobilenetv3
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import Conv2dNormActivation


class MobileNetV3Backbone(IntermediateLayerGetter):
    """Recommend to build mobilenet_v3_X with `dilated=True`

    Note that C2 has `output_stride = 8` and C3, C4, C5 have `output_stride = 16`
    """

    def __init__(self, model: mobilenetv3.MobileNetV3) -> None:
        features = model.features
        stage_indices = (
            [0]
            + [i for i, f in enumerate(features) if getattr(f, "_is_cn", False)]
            + [len(features) - 1]
        )
        return_layers = {str(stage): f"C{i}" for i, stage in enumerate(stage_indices)}
        super().__init__(features, return_layers)

    def layer_channels(self) -> dict[str, int]:
        num_channels: dict[str, int] = {}
        for name, module in self.named_children():
            if name not in self.return_layers:
                continue

            key = self.return_layers[name]
            if isinstance(module, Conv2dNormActivation):
                num_channels[key] = cast(nn.Conv2d, module[0]).out_channels
            elif isinstance(module, mobilenetv3.InvertedResidual):
                num_channels[key] = cast(nn.Conv2d, module.block[-1][0]).out_channels
            else:
                raise ValueError(f"Unknown block {key} of type {type(module)}")
        return num_channels
