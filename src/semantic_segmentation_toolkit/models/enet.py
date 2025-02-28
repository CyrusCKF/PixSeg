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


class ENet(nn.Module):
    """Implements ENet from [ENet: A Deep Neural Network Architecture for
    Real-Time Semantic Segmentation](https://arxiv.org/abs/1606.02147)"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.initial = ENetInitial()


class ENetInitial(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 13, 3, stride=2)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: Tensor):
        conv_out = self.conv(x)
        pool_out = self.conv(x)
        out = torch.cat([conv_out, pool_out], dim=1)
        return out


class ENetBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        projection_factor=4,
        dilation=1,
        dropout=0.01,
    ) -> None:
        super().__init__()
        inter_channels = in_channels // 4
        self.conv_modules = nn.ModuleList()
        self.conv_modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                nn.BatchNorm2d(inter_channels),
                nn.PReLU(),
            )
        )

        # initial conv
        nn.Conv2d(in_channels, inter_channels, 2, stride=2)
        conv_modules = [
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.PReLU(),
        ]

        # regular/dilated, full, asymmetric
        nn.Conv2d(
            inter_channels,
            inter_channels,
            3,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        nn.ConvTranspose2d(inter_channels, inter_channels, 3, padding=1, bias=False)
        [
            nn.Conv2d(
                inter_channels, inter_channels, (5, 1), padding=(2, 0), bias=False
            ),
            nn.BatchNorm2d(inter_channels),
            nn.PReLU(),
            nn.Conv2d(
                inter_channels, inter_channels, (1, 5), padding=(0, 2), bias=False
            ),
        ]
        [
            nn.BatchNorm2d(inter_channels),
            nn.PReLU(),
        ]

        # final conv
        [
            nn.Conv2d(inter_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.PReLU(),
        ]

        regularizer = nn.Dropout2d(dropout)

        # skipped connection
        nn.MaxPool2d(2)

        final_act = nn.PReLU()
