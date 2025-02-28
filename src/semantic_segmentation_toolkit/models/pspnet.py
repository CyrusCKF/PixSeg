from typing import Literal, Sequence

import torch
from torch import Tensor, nn
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F
from torchvision.models.resnet import *  # type: ignore
from torchvision.models.segmentation._utils import _SimpleSegmentationModel
from torchvision.models.segmentation.fcn import FCNHead

from ..datasets import CITYSCAPES_LABELS, VOC_LABELS
from .backbones import ResNetBackbone, replace_layer_name
from .model_registry import SegWeights, SegWeightsEnum, register_model


class PSPNet(_SimpleSegmentationModel):
    """Implements PSPNet from [Pyramid Scene Parsing
    Network](https://arxiv.org/abs/1612.01105)"""


class PSPHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pooling_sizes: Sequence[int] = (1, 2, 3, 6),
        pool_layer: Literal["avg", "max"] = "avg",
    ) -> None:
        super().__init__()
        self.poolings = nn.ModuleList()
        out_chan = in_channels // len(pooling_sizes)

        pool_nn = nn.AdaptiveAvgPool2d if pool_layer == "avg" else nn.AdaptiveMaxPool2d
        for size in pooling_sizes:
            mods = [
                pool_nn(size),
                nn.Conv2d(in_channels, out_chan, 1, bias=False),
                nn.BatchNorm2d(out_chan),
                nn.ReLU(),
            ]
            self.poolings.append(nn.Sequential(*mods))

        pyramid_channels = in_channels + out_chan * len(pooling_sizes)
        # self.bottleneck = nn.Sequential(
        #     nn.Conv2d(pyramid_channels, out_channels, 3, padding=1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(),
        # )
        self.head = FCNHead(pyramid_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        pools: list[Tensor] = []
        for pooling in self.poolings:
            pool = pooling(x)
            pools.append(F.interpolate(pool, x.shape[2:], mode="bilinear"))

        out = torch.cat([x, *pools], dim=1)
        # out = self.bottleneck(out)
        out = self.head(out)
        return out


class PSPNET_ResNet50_Weights(SegWeightsEnum):
    VOC2012 = SegWeights(
        "pspnet/pspnet_resnet50-voc2012-500x500-20250222.pth",
        VOC_LABELS,
        "Trained on PASCAL VOC 2012 dataset",
    )
    CITYSCAPES_FINE = SegWeights(
        "pspnet/pspnet_resnet50-cityscapes-512x1024-20250226.pth",
        CITYSCAPES_LABELS,
        "Trained on Cityscapes (fine) dataset",
    )
    DEFAULT = VOC2012


@register_model(weights_enum=PSPNET_ResNet50_Weights)
def pspnet_resnet50(
    num_classes: int | None = None,
    weights: PSPNET_ResNet50_Weights | str | None = None,
    progress: bool = True,
    aux_loss: bool = False,
    weights_backbone: ResNet50_Weights | str | None = ResNet50_Weights.DEFAULT,
):
    weights_model = PSPNET_ResNet50_Weights.resolve(weights)
    if num_classes is None:
        num_classes = 21 if weights_model is None else len(weights_model.labels)
    if weights_model is not None:
        weights_backbone = None
        if num_classes != len(weights_model.labels):
            raise ValueError(
                f"Model weights {weights_model} expect number of classes"
                f"={len(weights_model.labels)}, but got {num_classes}"
            )

    backbone_model = resnet50(weights=weights_backbone, progress=progress)
    backbone = ResNetBackbone(backbone_model)
    replace_layer_name(backbone, {-1: "out", -2: "aux"})

    channels = backbone.layer_channels()
    aux_classifier = FCNHead(channels["aux"], num_classes) if aux_loss else None
    classifier = PSPHead(channels["out"], num_classes)
    model = PSPNet(backbone, classifier, aux_classifier)

    if weights_model is not None:
        state_dict = load_state_dict_from_url(weights_model.url, progress=progress)
        model.load_state_dict(state_dict)
    return model
