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


# TODO clean up this section
class SFNet(nn.Module):
    """Implement SFNet from [Semantic Flow for Fast and Accurate Scene
    Parsing](https://arxiv.org/pdf/2002.10120)"""

    def __init__(
        self,
        num_classes: int,
        backbone: nn.Module,
        backbone_channels: dict[str, int],
        # variant="D",
        # head_type="v1",
        # skip="m1",
        # skip_num=48,
        fpn_dsn=False,
        # fa_type="spatial",
        global_context="ppm",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = UperNetAlignHead(
            list(backbone_channels.values())[-1],
            num_class=num_classes,
            fpn_inplanes=[64, 128, 256, 512],
            fpn_dim=128,
            fpn_dsn=fpn_dsn,
            global_context=global_context,
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        feature_maps: dict[str, Tensor] = self.backbone(x)
        outs: tuple[Tensor, list[Tensor]] = self.head(list(feature_maps.values()))
        # print("outs", outs[0].shape, [type(e) for e in outs[1]])
        main_out = F.interpolate(outs[0], size=x.shape[2:], mode="bilinear")
        return {"out": main_out}


class PSPModule(nn.Module):
    def __init__(
        self, features, out_features=512, sizes=(1, 2, 3, 6), norm_layer=nn.BatchNorm2d
    ):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList(
            [
                self._make_stage(features, out_features, size, norm_layer)
                for size in sizes
            ]
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                features + len(sizes) * out_features,
                out_features,
                kernel_size=1,
                padding=0,
                dilation=1,
                bias=False,
            ),
            norm_layer(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1),
        )

    def _make_stage(self, features, out_features, size, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = norm_layer(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [
            F.interpolate(
                input=stage(feats), size=(h, w), mode="bilinear", align_corners=True
            )
            for stage in self.stages
        ] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


class SPSPModule(nn.Module):
    """PSP with skip connection"""

    def __init__(
        self, features, out_features=512, sizes=(1, 2, 3, 6), norm_layer=nn.BatchNorm2d
    ):
        super(SPSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList(
            [
                self._make_stage(features, out_features, size, norm_layer)
                for size in sizes
            ]
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                out_features,
                out_features,
                kernel_size=1,
                padding=0,
                dilation=1,
                bias=False,
            ),
            norm_layer(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1),
        )
        self.down = nn.Sequential(
            nn.Conv2d(features, out_features, kernel_size=1, bias=False),
            norm_layer(out_features),
        )

    def _make_stage(self, features, out_features, size, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = norm_layer(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [
            F.interpolate(
                input=stage(feats), size=(h, w), mode="bilinear", align_corners=True
            )
            for stage in self.stages
        ]
        sum_feat = self.down(feats)

        for feat in priors:
            sum_feat = sum_feat + feat

        bottle = self.bottleneck(sum_feat)
        return bottle


class UperNetAlignHead(nn.Module):
    def __init__(
        self,
        inplane,
        num_class,
        norm_layer=nn.BatchNorm2d,
        fpn_inplanes=[256, 512, 1024, 2048],
        fpn_dim=256,
        conv3x3_type="conv",
        fpn_dsn=False,
        global_context="ppm",
    ):
        super(UperNetAlignHead, self).__init__()
        if global_context == "ppm":
            self.ppm = PSPModule(inplane, norm_layer=norm_layer, out_features=fpn_dim)
        elif global_context == "sppm":
            self.ppm = SPSPModule(
                inplane, sizes=(1, 2, 4), norm_layer=norm_layer, out_features=fpn_dim
            )

        self.fpn_dsn = fpn_dsn
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    norm_layer(fpn_dim),
                    nn.ReLU(inplace=False),
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        self.fpn_out_align = []
        self.dsn = []
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(
                nn.Sequential(
                    conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
                )
            )

            if conv3x3_type == "conv":
                self.fpn_out_align.append(
                    FlowAlignmentModule(inplane=fpn_dim, outplane=fpn_dim // 2)
                )

            if self.fpn_dsn:
                self.dsn.append(
                    nn.Sequential(
                        nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1),
                        norm_layer(fpn_dim),
                        nn.ReLU(),
                        nn.Dropout2d(0.1),
                        nn.Conv2d(
                            fpn_dim,
                            num_class,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True,
                        ),
                    )
                )

        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.fpn_out_align = nn.ModuleList(self.fpn_out_align)

        if self.fpn_dsn:
            self.dsn = nn.ModuleList(self.dsn)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1),
        )

    def forward(self, conv_out):
        psp_out = self.ppm(conv_out[-1])

        f = psp_out
        fpn_feature_list = [psp_out]
        out = []
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)  # lateral branch
            f = self.fpn_out_align[i]([conv_x, f])
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))
            if self.fpn_dsn:
                out.append(self.dsn[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]

        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(
                nn.functional.interpolate(
                    fpn_feature_list[i],
                    output_size,
                    mode="bilinear",
                    align_corners=True,
                )
            )

        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        return x, out


class FlowAlignmentModule(nn.Module):
    """Quoted from the paper:

    > ... align feature maps of two adjacent levels by predicting a flow field
    inside the network
    """

    def __init__(self, inplane, outplane, kernel_size=3):
        super().__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(
            outplane * 2, 2, kernel_size=kernel_size, padding=1, bias=False
        )

    def forward(self, x):
        low_feature, high_feature = x
        h_feature_orign = high_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        high_feature = self.down_h(high_feature)
        high_feature = F.interpolate(
            high_feature, size=size, mode="bilinear", align_corners=True
        )
        flow = self.flow_make(torch.cat([high_feature, low_feature], 1))
        high_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return high_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output


def conv3x3_bn_relu(in_planes, out_planes, stride=1, normal_layer=nn.BatchNorm2d):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        normal_layer(out_planes),
        nn.ReLU(inplace=True),
    )


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


@register_model()
def sfnet_resnet18(
    num_classes: int | None = None,
    weights: str | None = None,
    progress: bool = True,
    aux_loss: bool = False,
    weights_backbone: ResNet18_Weights | str | None = ResNet18_Weights.DEFAULT,
) -> nn.Module:
    if weights is not None:
        raise NotImplementedError("Weights is not supported yet")
    _, weights_backbone, num_classes = _validate_weights_input(
        None, weights_backbone, num_classes
    )

    backbone_model = resnet18(weights=weights_backbone, progress=progress)
    backbone = ResNetBackbone(backbone_model)

    channels = backbone.layer_channels()
    model = SFNet(num_classes, backbone, channels)
    return model
