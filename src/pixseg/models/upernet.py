import torch
from torch import Tensor, nn
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F
from torchvision.models.resnet import (
    ResNet18_Weights,
    ResNet101_Weights,
    resnet18,
    resnet101,
)

from ..datasets import CITYSCAPES_LABELS, VOC_LABELS
from .backbones import ResNetBackbone
from .model_registry import SegWeights, SegWeightsEnum, register_model
from .model_utils import _generate_docstring, _validate_weights_input
from .pspnet import PyramidPoolingModule


class ConvNormAct(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UperNetHead(nn.Module):
    def __init__(
        self,
        out_channels: int,
        feature_channels: dict[str, int],
        fpn_channels: int,
    ):
        super().__init__()
        self.fpn_channels = fpn_channels
        fpn_keys = list(feature_channels.keys())[:-1]  # last layer is not needed
        self.fpn_entries = nn.ModuleDict(
            {k: ConvNormAct(feature_channels[k], fpn_channels, 1) for k in fpn_keys}
        )
        # expect the last feature have fpn_channels
        self.fpn_exits = nn.ModuleDict(
            {k: ConvNormAct(fpn_channels, fpn_channels, 3) for k in feature_channels}
        )

        self.final_conv = ConvNormAct(
            len(feature_channels) * fpn_channels, out_channels, 3
        )

    def forward(self, feature_maps: dict[str, Tensor]) -> Tensor:
        last_key = list(feature_maps.keys())[-1]
        assert (
            feature_maps[last_key].size(1) == self.fpn_channels
        ), f"Last feature should have {self.fpn_channels} channels"

        # stored layers in reverse order
        fpn_layers: dict[str, Tensor] = {last_key: feature_maps[last_key]}
        layer_acc = fpn_layers[last_key]  # accumulate layers
        for k in list(feature_maps.keys())[-2::-1]:  # skip last key
            feature_out = self.fpn_entries[k](feature_maps[k])
            layer_acc = F.interpolate(layer_acc, feature_out.shape[2:], mode="bilinear")
            layer_acc += feature_out
            fpn_layers[k] = layer_acc

        fpn_outs = {k: self.fpn_exits[k](v) for k, v in fpn_layers.items()}
        fpn_size = list(feature_maps.values())[0].shape[2:]
        fpn_outs = [
            F.interpolate(v, fpn_size, mode="bilinear", align_corners=True)
            for v in fpn_outs.values()
        ]

        fused_feature_map = torch.cat(fpn_outs, dim=1)
        out = self.final_conv(fused_feature_map)
        return out


class UperNet(nn.Module):
    """Implement UPerNet from [Unified Perceptual Parsing for Scene
    Understanding](https://arxiv.org/abs/1807.10221)

    This only uses the *object* head for semantic segmentation.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: nn.Module,
        backbone_channels: dict[str, int],
        fpn_channels=128,
    ) -> None:
        super().__init__()
        self.backbone = backbone

        in_channels = list(backbone_channels.values())[-1]
        self.neck = PyramidPoolingModule(
            in_channels,
            fpn_channels,
            (1, 2, 3, 6),
            pool_channels=fpn_channels,
            dropout=0.1,
            bottleneck_kernel_size=3,
        )

        self.head = UperNetHead(num_classes, backbone_channels, fpn_channels)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        feature_maps: dict[str, Tensor] = self.backbone(x)
        last_key = list(feature_maps.keys())[-1]
        feature_maps[last_key] = self.neck(feature_maps[last_key])
        head_out: Tensor = self.head(feature_maps)
        main_out = F.interpolate(head_out, size=x.shape[2:], mode="bilinear")
        return {"out": main_out}


class UPerNet_ResNet18_Weights(SegWeightsEnum):
    CITYSCAPES = SegWeights(
        "upernet/upernet_resnet18-cityscapes-512x1024.pth",
        CITYSCAPES_LABELS,
        "Trained on Cityscapes (fine) dataset",
    )
    SBD = SegWeights(
        "upernet/upernet_resnet18-sbd-500x500.pth",
        VOC_LABELS,
        "Trained on Semantic Boundaries Dataset (SBD)",
    )
    DEFAULT = CITYSCAPES


class UPerNet_ResNet101_Weights(SegWeightsEnum):
    CITYSCAPES = SegWeights(
        "upernet/upernet_resnet101-cityscapes-512x1024.pth",
        CITYSCAPES_LABELS,
        "Trained on Cityscapes (fine) dataset",
    )
    DEFAULT = CITYSCAPES


@_generate_docstring("Unified Perceptual Network Lite model with a ResNet-18 backbone")
@register_model()
def upernet_resnet18(
    num_classes: int | None = None,
    weights: UPerNet_ResNet18_Weights | str | None = None,
    progress: bool = True,
    weights_backbone: ResNet18_Weights | str | None = ResNet18_Weights.DEFAULT,
) -> UperNet:
    weights_model = UPerNet_ResNet18_Weights.resolve(weights)
    weights_model, weights_backbone, num_classes = _validate_weights_input(
        weights_model, weights_backbone, num_classes
    )

    backbone_model = resnet18(weights=weights_backbone, progress=progress)
    backbone = ResNetBackbone(backbone_model)
    channels = backbone.layer_channels()
    model = UperNet(num_classes, backbone, channels)

    if weights_model is not None:
        state_dict = load_state_dict_from_url(weights_model.url, progress=progress)
        model.load_state_dict(state_dict)
    return model


@_generate_docstring("Unified Perceptual Network Lite model with a ResNet-101 backbone")
@register_model()
def upernet_resnet101(
    num_classes: int | None = None,
    weights: UPerNet_ResNet101_Weights | str | None = None,
    progress: bool = True,
    weights_backbone: ResNet101_Weights | str | None = ResNet101_Weights.DEFAULT,
) -> UperNet:
    weights_model = UPerNet_ResNet101_Weights.resolve(weights)
    weights_model, weights_backbone, num_classes = _validate_weights_input(
        weights_model, weights_backbone, num_classes
    )

    backbone_model = resnet101(weights=weights_backbone, progress=progress)
    backbone = ResNetBackbone(backbone_model)
    channels = backbone.layer_channels()
    model = UperNet(num_classes, backbone, channels)

    if weights_model is not None:
        state_dict = load_state_dict_from_url(weights_model.url, progress=progress)
        model.load_state_dict(state_dict)
    return model
