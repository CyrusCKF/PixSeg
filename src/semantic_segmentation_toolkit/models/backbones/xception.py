from torch import Tensor, nn
from torch.hub import load_state_dict_from_url
from torchvision.models._utils import IntermediateLayerGetter

from ..model_registry import SegWeights, SegWeightsEnum


#####
# region Model
#####
class SeparableConv(nn.Sequential):
    """No intermediate activation as discussed in section 4.7"""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size=3, bias=False
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                padding=kernel_size // 2,
                groups=in_channels,
                bias=bias,
            ),
            nn.Conv2d(in_channels, out_channels, 1, bias=bias),
        )


class ResidualBlock(nn.Module):
    def __init__(
        self,
        num_channels: list[int],
        relu_at_start=True,
        pool_at_end=True,
        residual_conv=True,
    ) -> None:
        """
        Args:
            num_channels: List of channels. The first is used as input and last as output.
                Intermediate convolution layers are built accordingly.
            residual_conv: Whether to pass residuals with 1x1 convolution
        """
        super().__init__()
        self.num_channels = num_channels

        # define residual path first, so that converting weights is less trickier
        self.residual_branch = nn.Identity()
        if residual_conv:
            self.residual_branch = nn.Sequential(
                nn.Conv2d(num_channels[0], num_channels[-1], 1, stride=2, bias=False),
                nn.BatchNorm2d(num_channels[-1]),
            )

        main_branch = []
        for in_channels, out_channels in zip(num_channels[:-1], num_channels[1:]):
            main_branch += [
                nn.ReLU(),
                SeparableConv(in_channels, out_channels, 3),
                nn.BatchNorm2d(out_channels),
            ]
        if not relu_at_start:
            main_branch.pop(0)
        if pool_at_end:
            main_branch.append(nn.MaxPool2d(3, stride=2, padding=1))
        self.main_branch = nn.Sequential(*main_branch)

    def forward(self, x: Tensor):
        main_out = self.main_branch(x)
        residual_out = self.residual_branch(x)
        return main_out + residual_out


class Xception(nn.Module):
    """Implement Xception from [Xception: Deep Learning with Depthwise
    Separable Convolutions](https://arxiv.org/abs/1610.02357)"""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.entry_flow = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock([64, 128, 128]),
            ResidualBlock([128, 256, 256]),
            ResidualBlock([256, 728, 728]),
        )

        middle_blocks = [
            ResidualBlock([728, 728, 728, 728], pool_at_end=False, residual_conv=False)
            for _ in range(8)
        ]
        self.middle_flow = nn.Sequential(*middle_blocks)

        self.exit_flow = nn.Sequential(
            ResidualBlock([728, 728, 1024]),
            SeparableConv(1024, 1536),
            nn.BatchNorm2d(1536),
            nn.ReLU(),
            SeparableConv(1536, 2048),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x: Tensor):
        out = self.entry_flow(x)
        out = self.middle_flow(out)
        out = self.exit_flow(out)
        return out


class Xception_Weights(SegWeightsEnum):
    IMAGENET1K = SegWeights(
        "xception/xception_imagenet.pth",
        tuple([f"Imagenet {i}" for i in range(1000)]),
        "Weights transformed from keras",
    )
    DEFAULT = IMAGENET1K


def xception_original(
    num_classes: int | None = None,
    weights: Xception_Weights | str | None = None,
    progress: bool = True,
):
    weights_model = Xception_Weights.resolve(weights)
    if num_classes is None:
        num_classes = 1000 if weights_model is None else len(weights_model.labels)
    if weights_model is not None and num_classes != len(weights_model.labels):
        raise ValueError(
            f"Model weights {weights} expect number of classes"
            f"={len(weights_model.labels)}, but got {num_classes}"
        )
    model = Xception(num_classes)
    if weights_model is not None:
        state_dict = load_state_dict_from_url(weights_model.url, progress=progress)
        model.load_state_dict(state_dict)
    return model


#####
# region Backbone
#####


class XceptionBackbone(IntermediateLayerGetter):
    def __init__(self, model: Xception) -> None:
        """Extract each spatial layers before the next pooling happens"""
        layer_and_channels: dict[str, tuple[nn.Module, int]] = {
            "layer1": (
                nn.Sequential(*model.entry_flow[:6]),
                model.entry_flow[4].num_features,
            ),
            "layer2": (model.entry_flow[6], model.entry_flow[6].num_channels[-1]),  # type: ignore
            "layer3": (model.entry_flow[7], model.entry_flow[7].num_channels[-1]),  # type: ignore
            "layer4": (
                nn.Sequential(model.entry_flow[8], *model.middle_flow),
                model.middle_flow[-1].num_channels[-1],  # type: ignore
            ),
            "layer5": (
                nn.Sequential(*model.exit_flow[:7]),
                model.exit_flow[5].num_features,
            ),
        }  # type: ignore
        layers = nn.ModuleDict({k: v[0] for k, v in layer_and_channels.items()})
        super().__init__(layers, {k: k for k in layers.keys()})
        self._orig_layer_channels = {k: v[1] for k, v in layer_and_channels.items()}

    def layer_channels(self) -> dict[str, int]:
        num_channels: dict[str, int] = {
            self.return_layers[k]: v for k, v in self._orig_layer_channels.items()
        }
        return num_channels
