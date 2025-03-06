from typing import Literal, Sequence

import torch
from torch import Tensor, nn
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F
from torchvision.models.resnet import ResNet18_Weights, resnet18
from torchvision.models.segmentation.fcn import FCNHead

from ..datasets import CITYSCAPES_LABELS
from .backbones import ResNetBackbone, replace_layer_name
from .model_registry import SegWeights, SegWeightsEnum, register_model
from .model_utils import _validate_weights_input
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


#####
# region Context
#####


class ContextPyramid(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, pooling_sizes: Sequence[int]
    ):
        super().__init__()
        self.poolings = nn.ModuleList()
        for size in pooling_sizes:
            self.poolings.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(size),
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
            )

    def _aggregate_pools(self, original: Tensor, pools: list[Tensor]) -> Tensor:
        raise NotImplementedError()

    def forward(self, x: Tensor) -> Tensor:
        pools: list[Tensor] = []
        for pooling in self.poolings:
            pool = pooling(x)
            pools.append(F.interpolate(pool, x.shape[2:], mode="bilinear"))
        return self._aggregate_pools(x, pools)


class ContextPPM(ContextPyramid):
    """Similar to pyramid pooling module, but with minor differences"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pooling_sizes: Sequence[int] = (1, 2, 3, 6),
    ):
        super().__init__(in_channels, out_channels, pooling_sizes)
        bottlenect_in = in_channels + len(pooling_sizes) * out_channels
        self.bottleneck = nn.Sequential(
            ConvNormAct(bottlenect_in, out_channels, 1),
            nn.Dropout2d(0.1),
        )

    def _aggregate_pools(self, original: Tensor, pools: list[Tensor]) -> Tensor:
        feature_cat = torch.cat([original] + pools, dim=1)
        return self.bottleneck(feature_cat)


class ContextSumPPM(ContextPyramid):
    """Summing over pooling layers instead of concatenating them"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pooling_sizes: Sequence[int] = (1, 2, 4),
    ):
        super().__init__(in_channels, out_channels, pooling_sizes)
        self.bottleneck = nn.Sequential(
            ConvNormAct(out_channels, out_channels, 1),
            nn.Dropout2d(0.1),
        )
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def _aggregate_pools(self, original: Tensor, pools: list[Tensor]) -> Tensor:
        feature_sum = sum([self.down_conv(original)] + pools)
        return self.bottleneck(feature_sum)


#####
# region FAM
#####


def pair_grid(xs: Tensor, ys: Tensor):
    """Return a pairing grid of shape (X, Y, 2) given 1d tensors. Element at [i, j] is (y, x)"""
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack((grid_y, grid_x), dim=-1)


def flow_warp(x: Tensor, flow_field: Tensor) -> Tensor:
    """Wrap the input according to the flow

    Args:
        x (Tensor (N, C, H, W)): Input tensor
        flow_field (Tensor (N, 2, FH, FW)): Relative offset on a normalized grid
            of `x`, i.e. element at *[n, :, x, y]* decides how the coordinates of `x`
            after normalizing to range (-1, 1) should flow.
    """
    FH, FW = flow_field.shape[-2:]
    assert flow_field.size(1) == 2, "flow_field should only has 2 channels"

    # make normalized coordinate grid
    fh_space = torch.linspace(-1.0, 1.0, FH)
    fw_space = torch.linspace(-1.0, 1.0, FW)
    coord_grid = pair_grid(fh_space, fw_space)

    norm_field = flow_field.permute(0, 2, 3, 1) / torch.tensor([FW, FH])
    coord_grid = coord_grid + norm_field
    output = F.grid_sample(x, coord_grid, align_corners=True)
    return output


class FlowAlignmentModule(nn.Module):
    """Quoted from the paper:

    > ... align feature maps of two adjacent levels by predicting a flow field
    inside the network
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down_high = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.down_low = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.flow_make = nn.Conv2d(out_channels * 2, 2, 3, padding=1, bias=False)

    def forward(self, low_feature: Tensor, high_feature: Tensor):
        """low_feature has larger spatial dimension and finer features"""
        output_size = low_feature.shape[2:]
        low_out = self.down_low(low_feature)
        high_out = self.down_high(high_feature)
        high_out = F.interpolate(
            high_out, output_size, mode="bilinear", align_corners=True
        )
        flow = self.flow_make(torch.cat([high_out, low_out], 1))
        out = flow_warp(high_feature, flow)
        return out


# TODO sfnet v2 module
class AlignedModulev2(nn.Module):

    def __init__(self, inplane, outplane, kernel_size=3):
        super().__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(
            outplane * 2, 4, kernel_size=kernel_size, padding=1, bias=False
        )
        self.flow_gate = nn.Sequential(
            nn.Conv2d(outplane * 2, 1, kernel_size=kernel_size, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        low_feature, h_feature = x
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        l_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.interpolate(
            h_feature, size=size, mode="bilinear", align_corners=True
        )

        flow = self.flow_make(torch.cat([h_feature, l_feature], 1))
        flow_up, flow_down = flow[:, :2, :, :], flow[:, 2:, :, :]

        h_feature_warp = flow_warp(h_feature_orign, flow_up)
        l_feature_warp = flow_warp(low_feature, flow_down)

        flow_gates = self.flow_gate(torch.cat([h_feature, l_feature], 1))

        fuse_feature = h_feature_warp * flow_gates + l_feature_warp * (1 - flow_gates)

        return fuse_feature


class AlignedModulev2PoolingAtten(nn.Module):

    def __init__(self, inplane, outplane, kernel_size=3):
        super(AlignedModulev2PoolingAtten, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(
            outplane * 2, 4, kernel_size=kernel_size, padding=1, bias=False
        )
        self.flow_gate = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=kernel_size, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        low_feature, h_feature = x
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        l_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.interpolate(
            h_feature, size=size, mode="bilinear", align_corners=True
        )

        flow = self.flow_make(torch.cat([h_feature, l_feature], 1))
        flow_up, flow_down = flow[:, :2, :, :], flow[:, 2:, :, :]

        h_feature_warp = flow_warp(h_feature_orign, flow_up)
        l_feature_warp = flow_warp(low_feature, flow_down)

        h_feature_mean = torch.mean(h_feature, dim=1).unsqueeze(1)
        l_feature_mean = torch.mean(low_feature, dim=1).unsqueeze(1)
        h_feature_max = torch.max(h_feature, dim=1)[0].unsqueeze(1)
        l_feature_max = torch.max(low_feature, dim=1)[0].unsqueeze(1)

        flow_gates = self.flow_gate(
            torch.cat([h_feature_mean, l_feature_mean, h_feature_max, l_feature_max], 1)
        )

        fuse_feature = h_feature_warp * flow_gates + l_feature_warp * (1 - flow_gates)

        return fuse_feature


#####
# region Head
#####


class SFNetHead(nn.Module):
    def __init__(
        self,
        out_channels: int,
        feature_channels: dict[str, int],
        fpn_channels=256,
        enable_dsn=False,
    ):
        super().__init__()
        fpn_keys = list(feature_channels.keys())[:-1]  # last layer is not needed
        self.fpn_ins = nn.ModuleDict()
        self.fpn_outs = nn.ModuleDict()
        self.fams = nn.ModuleDict()
        for k in fpn_keys:
            self.fpn_ins[k] = ConvNormAct(feature_channels[k], fpn_channels, 1)
            self.fpn_outs[k] = ConvNormAct(fpn_channels, fpn_channels, 3)
            self.fams[k] = FlowAlignmentModule(
                in_channels=fpn_channels, out_channels=fpn_channels // 2
            )

        self.dsns = None
        if enable_dsn:
            self.dsns = nn.ModuleDict()
            for k in fpn_keys:
                self.dsns[k] = nn.Sequential(
                    ConvNormAct(fpn_channels, fpn_channels, 3),
                    nn.Dropout2d(0.1),
                    nn.Conv2d(fpn_channels, out_channels, kernel_size=1, bias=True),
                )

        self.final_conv = nn.Sequential(
            ConvNormAct(len(feature_channels) * fpn_channels, fpn_channels, 3),
            nn.Conv2d(fpn_channels, out_channels, kernel_size=1),
        )

    def forward(
        self, feature_maps: dict[str, Tensor]
    ) -> tuple[Tensor, list[Tensor] | None]:
        last_feature = list(feature_maps.values())[-1]
        layer_outs: dict[str, Tensor] = {}
        layer_acc = last_feature
        reverse_fpn_keys = list(feature_maps.keys())[-2::-1]  # skip last key
        for k in reverse_fpn_keys:
            feature = feature_maps[k]
            feature_out = self.fpn_ins[k](feature)
            fam_out = self.fams[k](feature_out, layer_acc)
            layer_acc = feature_out + fam_out
            layer_outs[k] = layer_acc

        fpn_features = [last_feature]
        fpn_features += [self.fpn_outs[k](out) for k, out in layer_outs.items()]
        dsn_outs: list[Tensor] | None = None
        if self.dsns is not None:
            dsn_outs = [self.dsns[k](out) for k, out in layer_outs.items()]

        fpn_features.reverse()  # from 1/4 to 1/32 features
        # there can be FAMs during upsampling
        output_size = fpn_features[0].shape[2:]
        fusion_list = [fpn_features[0]]
        fusion_list += [
            F.interpolate(feat, output_size, mode="bilinear", align_corners=True)
            for feat in fpn_features[1:]
        ]

        fusion_out = torch.cat(fusion_list, 1)
        out = self.final_conv(fusion_out)
        return out, dsn_outs


# TODO sfnet v2 module
class UperNetAlignHeadV2(nn.Module):

    def __init__(
        self,
        num_class,
        norm_layer=nn.BatchNorm2d,
        fpn_inplanes=[256, 512, 1024, 2048],
        fpn_dim=256,
        fpn_dsn=False,
        fa_type="spatial",
    ):
        super(UperNetAlignHeadV2, self).__init__()

        self.fpn_dsn = fpn_dsn
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes:  # total 2 planes
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    norm_layer(fpn_dim),
                    nn.ReLU(inplace=False),
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out_align = []
        if fa_type == "spatial":
            self.flow_align_module = AlignedModulev2(
                inplane=fpn_dim, outplane=fpn_dim // 2
            )
        elif fa_type == "spatial_atten":
            self.flow_align_module = AlignedModulev2PoolingAtten(
                inplane=fpn_dim, outplane=fpn_dim // 2
            )

        self.fpn_out_align = nn.ModuleList(self.fpn_out_align)

        if self.fpn_dsn:
            self.dsn = nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1),
                norm_layer(fpn_dim),
                nn.ReLU(),
                nn.Dropout2d(0.1),
                nn.Conv2d(
                    fpn_dim, num_class, kernel_size=1, stride=1, padding=0, bias=True
                ),
            )

        self.conv_last = nn.Sequential(
            ConvNormAct(fpn_dim * 2, fpn_dim, 3),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1),
        )

    def forward(self, conv_out):

        # p2, p3, p4, p5(ppm)
        p2 = conv_out[0]
        p4 = conv_out[2]

        # print("p2", p2.shape, self.fpn_in[0][0].weight.shape)
        p2 = self.fpn_in[0](p2)
        p4 = self.fpn_in[1](p4)

        fusion_out = self.flow_align_module([p2, conv_out[-1]])
        output_size = fusion_out.size()[2:]

        p4 = nn.functional.interpolate(
            p4, output_size, mode="bilinear", align_corners=True
        )

        x = self.conv_last(torch.cat([fusion_out, p4], dim=1))

        return x, []


#####
# region SFNet
#####


class SFNet(nn.Module):
    """Implement SFNet from [Semantic Flow for Fast and Accurate Scene
    Parsing](https://arxiv.org/pdf/2002.10120)"""

    def __init__(
        self,
        num_classes: int,
        backbone: nn.Module,
        backbone_channels: dict[str, int],
        head_type: Literal["v1", "v2"] = "v1",
        enable_dsn=False,
        fpn_channels=128,
        fa_type="spatial",
        context_type: Literal["ppm", "sppm"] = "ppm",
    ) -> None:
        super().__init__()
        self.backbone = backbone

        in_channels = list(backbone_channels.values())[-1]
        if context_type == "ppm":
            self.neck = ContextPPM(in_channels, fpn_channels)
        elif context_type == "sppm":
            self.neck = ContextSumPPM(in_channels, fpn_channels)

        if head_type == "v2":
            # print("backbone_channels", backbone_channels)
            self.head = UperNetAlignHeadV2(
                num_class=num_classes,
                fpn_inplanes=[64, 256, 512],
                fpn_dim=fpn_channels,
                fpn_dsn=enable_dsn,
                fa_type=fa_type,
            )
        else:
            self.head = SFNetHead(
                out_channels=num_classes,
                feature_channels=backbone_channels,
                fpn_channels=fpn_channels,
                enable_dsn=enable_dsn,
            )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        feature_maps: dict[str, Tensor] = self.backbone(x)
        last_key = list(feature_maps.keys())[-1]
        feature_maps[last_key] = self.neck(feature_maps[last_key])

        if isinstance(self.head, UperNetAlignHeadV2):
            head_outs: tuple[Tensor, list[Tensor] | None] = self.head(
                list(feature_maps.values())
            )
        else:
            head_outs: tuple[Tensor, list[Tensor] | None] = self.head(feature_maps)
        # TODO use dsn
        # print("outs", outs[0].shape, [type(e) for e in outs[1]])
        main_out = F.interpolate(head_outs[0], size=x.shape[2:], mode="bilinear")
        return {"out": main_out}


@register_model()
def sfnet_resnet18(
    num_classes: int | None = None,
    weights: str | None = None,
    progress: bool = True,
    aux_loss: bool = False,
    weights_backbone: ResNet18_Weights | str | None = ResNet18_Weights.DEFAULT,
    **kwargs,
) -> SFNet:
    """See :class:`SFNet` for supported kwargs"""
    if weights is not None:
        raise NotImplementedError("Weights is not supported yet")
    _, weights_backbone, num_classes = _validate_weights_input(
        None, weights_backbone, num_classes
    )

    backbone_model = resnet18(weights=weights_backbone, progress=progress)
    backbone = ResNetBackbone(backbone_model)

    channels = backbone.layer_channels()
    kwargs["enable_dsn"] = aux_loss
    model = SFNet(num_classes, backbone, channels, **kwargs)
    return model
