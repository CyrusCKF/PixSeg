import torch
from torch import Tensor, nn
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F
from torchvision.models.resnet import ResNet18_Weights, resnet18
from torchvision.models.segmentation.fcn import FCNHead

from .backbones import ResNetBackbone, replace_layer_name
from .model_registry import register_model


def _conv_norm_act(
    in_channels: int, out_channels: int, kernel_size: int, stride: int = 1
) -> list[nn.Module]:
    return [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            stride=stride,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    ]


class BiSeNet(nn.Module):
    """Implement BiSeNet from [BiSeNet: Bilateral Segmentation Network for
    Real-time Semantic Segmentation](https://arxiv.org/abs/1808.00897)"""

    def __init__(
        self,
        num_classes: int,
        backbone: nn.Module,
        backbone_channels: dict[str, int],
        path_channels: int = 128,
        use_aux=False,
    ) -> None:
        """
        Args:
            backbone: Return features which must contain keys "16x" and "32x"
            backbone_channels: Number of channels in "16x" and "32x"
            path_channels: The output channels of spatial path and context path
        """
        super().__init__()
        self.backbone = backbone
        self.context_path_stage = ContextPathStage(backbone_channels, path_channels)
        self.spatial_path = SpatialPath(3, path_channels)
        self.ffm = FeatureFusionModule(path_channels * 2, path_channels * 2)

        self.head = FCNHead(path_channels * 2, num_classes)
        self.aux_head = FCNHead(path_channels, num_classes) if use_aux else None

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        spatial_out = self.spatial_path(x)
        print(f"{spatial_out.shape=}")
        feature_maps: dict[str, Tensor] = self.backbone(x)
        context_out = self.context_path_stage(feature_maps, spatial_out.shape[2:])
        print(f"{context_out.shape=}")
        combined_out = torch.cat([spatial_out, context_out], dim=1)
        ffm_out = self.ffm(combined_out)
        print(f"{ffm_out.shape=}")

        main_out = self.head(ffm_out)
        main_out = F.interpolate(main_out, x.shape[2:])
        out = {"out": main_out}
        if self.aux_head is not None:
            aux_out = self.aux_head(context_out)
            aux_out = F.interpolate(aux_out, x.shape[2:])
            out["aux"] = aux_out
        return out


class ContextPathStage(nn.Module):
    def __init__(self, feature_channels: dict[str, int], out_channels: int):
        """feature_channels must contain keys "16x" and "32x"""
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.arm_16x = AttentionRefinementModule(feature_channels["16x"], out_channels)
        self.arm_32x = AttentionRefinementModule(feature_channels["32x"], out_channels)
        self.conv_16x = nn.Sequential(*_conv_norm_act(out_channels, out_channels, 3))
        self.conv_32x = nn.Sequential(*_conv_norm_act(out_channels, out_channels, 3))
        self.conv_global = nn.Sequential(
            *_conv_norm_act(feature_channels["32x"], out_channels, 1)
        )

    def forward(self, feature_maps: dict[str, Tensor], output_size: list[int]):
        """feature_maps must contain keys "16x" and "32x"""
        feature_16x, feature_32x = feature_maps["16x"], feature_maps["32x"]

        out = self.global_pool(feature_32x)
        out = self.conv_global(out)

        out = self.arm_32x(feature_32x) + out
        out = F.interpolate(out, feature_16x.shape[2:])
        out = self.conv_32x(out)

        out = self.arm_16x(feature_16x) + out
        out = F.interpolate(out, output_size)
        out = self.conv_16x(out)

        return out


class SpatialPath(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        inter_channels = out_channels // 2
        super().__init__(
            *_conv_norm_act(in_channels, inter_channels, 7, 2),
            *_conv_norm_act(inter_channels, inter_channels, 3, 2),
            *_conv_norm_act(inter_channels, inter_channels, 3, 2),
            *_conv_norm_act(inter_channels, out_channels, 1, 1),
        )


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(*_conv_norm_act(in_channels, out_channels, 3, 1))
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor):
        features = self.conv(x)
        weight = self.attention(features)
        out = features * weight
        return out


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(*_conv_norm_act(in_channels, out_channels, 1))
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor):
        features = self.conv(x)
        weight = self.attention(features)
        out = features + features * weight
        return out


@register_model()
def bisenet_resnet18(
    num_classes: int | None = None,
    weights: str | None = None,
    progress: bool = True,
    aux_loss: bool = False,
    weights_backbone: ResNet18_Weights | str | None = ResNet18_Weights.DEFAULT,
) -> nn.Module:
    if weights is not None:
        raise NotImplementedError("Weights is not supported yet")
    if num_classes is None:
        num_classes = 21

    backbone_model = resnet18(weights=weights_backbone, progress=progress)
    backbone = ResNetBackbone(backbone_model)
    replace_layer_name(backbone, {-2: "16x", -1: "32x"})

    channels = backbone.layer_channels()
    model = BiSeNet(20, backbone, channels, use_aux=aux_loss)
    return model
