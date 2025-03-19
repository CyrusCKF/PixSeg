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
from .sfnet import ConvNormAct, SFNet, flow_warp


class FlowAlignmentModuleV2(nn.Module):
    """Second version of FAM, which incorporates *flow gate* to interpolate
    the low and high features, and optionally pooling them"""

    def __init__(self, in_channels: int, out_channels: int, with_pooling=False):
        """
        Args:
            with_pooling: If `True`, flow gate is calculated after pooling the
                high feature and low feature.
        """
        super().__init__()
        self.down_high = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.down_low = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.flow_make = nn.Conv2d(out_channels * 2, 4, 3, padding=1, bias=False)

        self.with_pooling = with_pooling
        flow_gate_in = 4 if with_pooling else out_channels * 2
        self.flow_gate = nn.Sequential(
            nn.Conv2d(flow_gate_in, 1, 3, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, low_feature: Tensor, high_feature: Tensor):
        """low_feature has larger spatial dimension and finer features"""
        output_size = low_feature.shape[2:]
        low_out: Tensor = self.down_low(low_feature)
        high_out: Tensor = self.down_high(high_feature)
        high_out = F.interpolate(
            high_out, output_size, mode="bilinear", align_corners=True
        )
        joined_feature_out = torch.cat([high_out, low_out], dim=1)

        flow = self.flow_make(joined_feature_out)
        flow_up, flow_down = flow[:, :2, :, :], flow[:, 2:, :, :]
        high_warp = flow_warp(high_feature, flow_up)
        low_warp = flow_warp(low_feature, flow_down)

        if self.with_pooling:
            feature_pools = [
                high_out.mean(dim=1, keepdim=True),
                low_out.mean(dim=1, keepdim=True),
                high_out.max(dim=1, keepdim=True)[0],
                low_out.max(dim=1, keepdim=True)[0],
            ]
            joined_feature_pool = torch.cat(feature_pools, dim=1)
            gate = self.flow_gate(joined_feature_pool)
        else:
            gate = self.flow_gate(joined_feature_out)
        fuse_feature = high_warp.lerp(low_warp, gate)
        return fuse_feature


class SFNetLiteHead(nn.Module):
    """Lightweight version of :class:`SFNetHead`, which flow aligns the first
    and last feature maps, then concatenate with the third map as the final features."""

    def __init__(
        self,
        out_channels: int,
        feature_channels: dict[str, int],
        fpn_channels=256,
        fam_pooling=False,
    ):
        super().__init__()
        # only first and third features in fpn
        self.fpn_in_0 = ConvNormAct(list(feature_channels.values())[0], fpn_channels, 1)
        self.fpn_in_2 = ConvNormAct(list(feature_channels.values())[2], fpn_channels, 1)
        self.fam = FlowAlignmentModuleV2(
            fpn_channels, fpn_channels // 2, with_pooling=fam_pooling
        )

        self.final_conv = nn.Sequential(
            ConvNormAct(fpn_channels * 2, fpn_channels, 3),
            nn.Conv2d(fpn_channels, out_channels, kernel_size=1),
        )

    def forward(self, feature_maps: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Only use the first, third and last feature maps"""
        feature_0 = list(feature_maps.values())[0]
        feature_2 = list(feature_maps.values())[2]
        feature_last = list(feature_maps.values())[-1]

        feature_0_out = self.fpn_in_0(feature_0)
        fusion_out = self.fam(feature_0_out, feature_last)
        output_size = fusion_out.shape[2:]

        feature_2_out = self.fpn_in_2(feature_2)
        feature_2_out = F.interpolate(
            feature_2_out, output_size, mode="bilinear", align_corners=True
        )

        combined = torch.cat([fusion_out, feature_2_out], dim=1)
        out = self.final_conv(combined)
        return out, None


class SFNetLite(SFNet):
    """It is published as SFNet-Lite, but called SFNetV2 in code"""

    def __init__(
        self,
        num_classes: int,
        backbone: nn.Module,
        backbone_channels: dict[str, int],
        fpn_channels=128,
        fam_pooling=False,
        enable_dsn=False,
    ) -> None:
        super().__init__(
            num_classes, backbone, backbone_channels, fpn_channels, enable_dsn
        )
        self.head = SFNetLiteHead(
            num_classes,
            backbone_channels,
            fpn_channels=fpn_channels,
            fam_pooling=fam_pooling,
        )


class SFNetLite_ResNet18_Weights(SegWeightsEnum):
    CITYSCAPES = SegWeights(
        "sfnet-lite/sfnet_lite_resnet18-cityscapes-512x1024.pth",
        CITYSCAPES_LABELS,
        "Trained on Cityscapes (fine) dataset",
    )
    SBD = SegWeights(
        "sfnet-lite/sfnet_lite_resnet18-sbd-500x500.pth",
        VOC_LABELS,
        "Trained on Semantic Boundaries Dataset (SBD)",
    )
    DEFAULT = CITYSCAPES


class SFNetLite_ResNet101_Weights(SegWeightsEnum):
    CITYSCAPES = SegWeights(
        "sfnet-lite/sfnet_lite_resnet101-cityscapes-512x1024.pth",
        CITYSCAPES_LABELS,
        "Trained on Cityscapes (fine) dataset",
    )
    DEFAULT = CITYSCAPES


@_generate_docstring("Semantic Flow Network Lite model with a ResNet-18 backbone")
@register_model()
def sfnet_lite_resnet18(
    num_classes: int | None = None,
    weights: SFNetLite_ResNet18_Weights | str | None = None,
    progress: bool = True,
    weights_backbone: ResNet18_Weights | str | None = ResNet18_Weights.DEFAULT,
    **kwargs,
) -> SFNetLite:
    """See :class:`SFNetLite` for supported kwargs"""
    weights_model = SFNetLite_ResNet18_Weights.resolve(weights)
    weights_model, weights_backbone, num_classes = _validate_weights_input(
        weights_model, weights_backbone, num_classes
    )

    backbone_model = resnet18(weights=weights_backbone, progress=progress)
    backbone = ResNetBackbone(backbone_model)
    channels = backbone.layer_channels()
    model = SFNetLite(num_classes, backbone, channels, **kwargs)

    if weights_model is not None:
        state_dict = load_state_dict_from_url(weights_model.url, progress=progress)
        model.load_state_dict(state_dict)
    return model


@_generate_docstring("Semantic Flow Network Lite model with a ResNet-101 backbone")
@register_model()
def sfnet_lite_resnet101(
    num_classes: int | None = None,
    weights: SFNetLite_ResNet101_Weights | str | None = None,
    progress: bool = True,
    weights_backbone: ResNet101_Weights | str | None = ResNet101_Weights.DEFAULT,
    **kwargs,
) -> SFNetLite:
    """See :class:`SFNetLite` for supported kwargs"""
    weights_model = SFNetLite_ResNet18_Weights.resolve(weights)
    weights_model, weights_backbone, num_classes = _validate_weights_input(
        weights_model, weights_backbone, num_classes
    )

    backbone_model = resnet101(weights=weights_backbone, progress=progress)
    backbone = ResNetBackbone(backbone_model)

    channels = backbone.layer_channels()
    model = SFNetLite(num_classes, backbone, channels, **kwargs)

    if weights_model is not None:
        state_dict = load_state_dict_from_url(weights_model.url, progress=progress)
        model.load_state_dict(state_dict)
    return model
