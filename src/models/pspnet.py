import sys
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F
from torchvision.models.resnet import *  # type: ignore
from torchvision.models.segmentation._utils import _SimpleSegmentationModel
from torchvision.models.segmentation.fcn import FCNHead

sys.path.append(str((Path(__file__) / "..").resolve()))
from backbones import ResNetBackbone, replace_layer_name
from model_api import SegWeights, SegWeightsEnum
from model_registry import register_model

sys.path.append(str((Path(__file__) / "../../..").resolve()))
from src.datasets.pytorch_datasets import VOC_LABELS
from src.utils.transform import SegmentationAugment


# TODO use custom implementation of _SimpleSegmentationModel
class PSPNet(_SimpleSegmentationModel):
    """Implements PSPNet model from `Pyramid Scene Parsing Network
    <https://arxiv.org/abs/1612.01105>`_"""


class PSPHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pooling_sizes: Sequence[int] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()
        self.poolings = nn.ModuleList()
        out_chan = in_channels // len(pooling_sizes)
        for size in pooling_sizes:
            mods = [
                nn.AdaptiveAvgPool2d(size),  # TODO support max pool
                nn.Conv2d(in_channels, out_chan, 1, bias=False),
                nn.BatchNorm2d(out_chan),
                nn.ReLU(),
            ]
            self.poolings.append(nn.Sequential(*mods))

        pyramid_channels = in_channels + out_chan * len(pooling_sizes)
        self.head = FCNHead(pyramid_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        pools: list[Tensor] = []
        for pooling in self.poolings:
            pool = pooling(x)
            pools.append(F.interpolate(pool, x.shape[-2:], mode="bilinear"))

        feature_and_pools = torch.cat([x, *pools], dim=1)
        out = self.head(feature_and_pools)
        return out


@register_model()
def pspnet_resnet50(
    num_classes: int | None = None,
    weights: str | None = None,
    progress: bool = True,
    aux_loss: bool = False,
    weights_backbone: ResNet50_Weights | str | None = ResNet50_Weights.DEFAULT,
):
    if weights is not None:
        raise NotImplementedError("Weights is not supported yet")
    if num_classes is None:
        num_classes = 21

    backbone_model = resnet50(weights=weights_backbone, progress=progress)
    backbone = ResNetBackbone(backbone_model)
    replace_layer_name(backbone, {-1: "out", -2: "aux"})

    channels = backbone.layer_channels()
    aux_classifier = FCNHead(channels["aux"], num_classes) if aux_loss else None
    classifier = PSPHead(channels["out"], num_classes)
    return PSPNet(backbone, classifier, aux_classifier)
