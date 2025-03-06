from typing import Sequence

import torch
from torch import Tensor, nn
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F
from torchvision.models.resnet import ResNet50_Weights, resnet50
from torchvision.models.segmentation._utils import _SimpleSegmentationModel
from torchvision.models.segmentation.fcn import FCNHead

from ..datasets import CITYSCAPES_LABELS, VOC_LABELS
from .backbones import ResNetBackbone, replace_layer_name
from .model_registry import SegWeights, SegWeightsEnum, register_model
from .model_utils import _validate_weights_input


class PyramidPoolingModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pooling_sizes: Sequence[int] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()
        self.poolings = nn.ModuleList()
        pool_channels = in_channels // len(pooling_sizes)

        for size in pooling_sizes:
            mods = [
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, pool_channels, 1, bias=False),
                nn.BatchNorm2d(pool_channels),
                nn.ReLU(),
            ]
            self.poolings.append(nn.Sequential(*mods))

        pyramid_channels = in_channels + pool_channels * len(pooling_sizes)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(pyramid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        pools: list[Tensor] = []
        for pooling in self.poolings:
            pool = pooling(x)
            pools.append(F.interpolate(pool, x.shape[2:], mode="bilinear"))

        out = torch.cat([x, *pools], dim=1)
        out = self.bottleneck(out)
        return out


class PSPHead(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pooling_sizes: Sequence[int] = (1, 2, 3, 6),
    ) -> None:
        super().__init__(
            PyramidPoolingModule(in_channels, out_channels, pooling_sizes),
            FCNHead(out_channels, out_channels),
        )


class PSPNet(_SimpleSegmentationModel):
    """Implements PSPNet from [Pyramid Scene Parsing
    Network](https://arxiv.org/abs/1612.01105)"""


class PSPNET_ResNet50_Weights(SegWeightsEnum):
    VOC2012 = SegWeights(
        "pspnet/pspnet_resnet50-voc2012-500x500-20250222.pth",
        VOC_LABELS,
        "Trained on PASCAL VOC 2012 dataset",
    )
    CITYSCAPES_FINE = SegWeights(
        "pspnet/pspnet_resnet50-cityscapes-512x1024-20250304.pth",
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
    weights_model, weights_backbone, num_classes = _validate_weights_input(
        weights_model, weights_backbone, num_classes
    )

    backbone_model = resnet50(
        weights=weights_backbone,
        progress=progress,
        replace_stride_with_dilation=[False, True, True],
    )
    backbone = ResNetBackbone(backbone_model)
    replace_layer_name(backbone, {-1: "out", -2: "aux"})

    channels = backbone.layer_channels()
    classifier = PSPHead(channels["out"], num_classes)
    aux_classifier = FCNHead(channels["aux"], num_classes) if aux_loss else None
    model = PSPNet(backbone, classifier, aux_classifier)

    if weights_model is not None:
        state_dict = load_state_dict_from_url(weights_model.url, progress=progress)
        model.load_state_dict(state_dict)
    return model
