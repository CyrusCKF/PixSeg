import torch
from torch import Tensor, nn
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F

from ..datasets import CITYSCAPES_LABELS
from .model_registry import SegWeights, SegWeightsEnum, register_model
from .model_utils import _validate_weights_input


def _pad_to_even_size(x: Tensor, value):
    # pad so that input is even for each downsample; otherwise unpooling is messy to deal with
    pad_size: list[int] = []
    for size in x.shape[-1:1:-1]:  # get size in reverse order to the third dim
        to_pad = [0, 0] if size % 2 == 0 else [0, 1]
        pad_size.extend(to_pad)
    return F.pad(x, pad_size, value=value)


class ENetInitial(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels - 3, 3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.norm_act = nn.Sequential(nn.BatchNorm2d(out_channels), nn.PReLU())

    def forward(self, x: Tensor):
        conv_out = self.conv(x)
        pool_out = self.pool(_pad_to_even_size(x, -torch.inf))
        out = torch.cat([conv_out, pool_out], dim=1)
        out = self.norm_act(out)
        return out


class ENetBottleneck(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, projection_factor=4, dropout=0.01
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.projection_factor = projection_factor
        self.dropout = dropout
        self.inter_channels = in_channels // projection_factor
        convs = self._make_convs(in_channels, self.inter_channels, out_channels)

        self.main_modules = nn.ModuleList()
        for i, conv in enumerate(convs):
            self.main_modules.append(conv)
            if i != len(convs) - 1:
                assert isinstance(conv, (nn.Conv2d, nn.ConvTranspose2d))
                out_chan = conv.out_channels
                self.main_modules += [nn.BatchNorm2d(out_chan), nn.PReLU()]
        self.main_modules.append(nn.Dropout2d(dropout))

        self.final_act = nn.PReLU()

    def _make_convs(self, in_chan, inter_chan, out_chan) -> list[nn.Module]:
        raise NotImplementedError("Please implement this")


class ENetUpsampleBottleneck(ENetBottleneck):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv_before_unpool = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        self.unpool = nn.MaxUnpool2d(2)

    def _make_convs(self, in_chan, inter_chan, out_chan) -> list[nn.Module]:
        return [
            nn.Conv2d(in_chan, inter_chan, 1, bias=False),
            nn.ConvTranspose2d(inter_chan, inter_chan, 2, stride=2, bias=False),
            nn.Conv2d(inter_chan, out_chan, 1, bias=False),
        ]

    def forward(self, x: Tensor, pooling_indices: Tensor, output_size: list[int]):
        main_out = x
        for main in self.main_modules:
            if isinstance(main, nn.ConvTranspose2d):
                main_out = main(main_out, output_size=output_size)
            else:
                main_out = main(main_out)
        pool_out = self.conv_before_unpool(x)
        pool_out = self.unpool(pool_out, pooling_indices, output_size)
        out = main_out + pool_out
        out = self.final_act(out)
        return out


class ENetDownsampleBottleneck(ENetBottleneck):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pool = nn.MaxPool2d(2, return_indices=True)

    def _make_convs(self, in_chan, inter_chan, out_chan) -> list[nn.Module]:
        return [
            nn.Conv2d(in_chan, inter_chan, 2, stride=2, bias=False),
            nn.Conv2d(inter_chan, inter_chan, 3, padding=1, bias=False),
            nn.Conv2d(inter_chan, out_chan, 1, bias=False),
        ]

    def forward(self, x: Tensor):
        pool_out, indices = self.pool(x)
        main_out = x
        for main in self.main_modules:
            main_out = main(main_out)

        out = main_out
        out[:, : pool_out.size(1)] += pool_out  # same as zero padded
        out = self.final_act(out)
        return out, indices


class ENetRegularBottleneck(ENetBottleneck):
    def __init__(self, *args, dilation=1, asymmetric=False, **kwargs) -> None:
        self.dilation = dilation
        self.asymmetric = asymmetric
        super().__init__(*args, **kwargs)

    def _make_convs(self, in_chan, inter_chan, out_chan) -> list[nn.Module]:
        convs: list[nn.Module] = [nn.Conv2d(in_chan, inter_chan, 1, bias=False)]
        if self.asymmetric:
            convs += [
                nn.Conv2d(inter_chan, inter_chan, (5, 1), padding=(2, 0), bias=False),
                nn.Conv2d(inter_chan, inter_chan, (1, 5), padding=(0, 2), bias=False),
            ]
        else:
            di = self.dilation
            convs.append(
                nn.Conv2d(
                    inter_chan, inter_chan, 3, dilation=di, padding=di, bias=False
                )
            )
        convs.append(nn.Conv2d(inter_chan, out_chan, 1, bias=False))
        return convs

    def forward(self, x: Tensor):
        main_out = x
        for main in self.main_modules:
            main_out = main(main_out)
        out = main_out + x
        out = self.final_act(out)
        return out


class ENet(nn.Module):
    """Implement ENet from [ENet: A Deep Neural Network Architecture for
    Real-Time Semantic Segmentation](https://arxiv.org/abs/1606.02147)"""

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.initial = ENetInitial(3, 16)

        self.section1_down = ENetDownsampleBottleneck(16, 64)
        self.section1_convs = nn.Sequential(
            *[ENetRegularBottleneck(64, 64) for _ in range(4)]
        )

        self.section2_down = ENetDownsampleBottleneck(64, 128)
        self.section2_convs = nn.Sequential(
            ENetRegularBottleneck(128, 128),
            ENetRegularBottleneck(128, 128, dilation=2),
            ENetRegularBottleneck(128, 128, asymmetric=True),
            ENetRegularBottleneck(128, 128, dilation=4),
            ENetRegularBottleneck(128, 128),
            ENetRegularBottleneck(128, 128, dilation=8),
            ENetRegularBottleneck(128, 128, asymmetric=True),
            ENetRegularBottleneck(128, 128, dilation=16),
        )

        self.section3_convs = nn.Sequential(
            ENetRegularBottleneck(128, 128),
            ENetRegularBottleneck(128, 128, dilation=2),
            ENetRegularBottleneck(128, 128, asymmetric=True),
            ENetRegularBottleneck(128, 128, dilation=4),
            ENetRegularBottleneck(128, 128),
            ENetRegularBottleneck(128, 128, dilation=8),
            ENetRegularBottleneck(128, 128, asymmetric=True),
            ENetRegularBottleneck(128, 128, dilation=16),
        )

        self.section4_up = ENetUpsampleBottleneck(128, 64)
        self.section4_convs = nn.Sequential(
            *[ENetRegularBottleneck(64, 64) for _ in range(2)]
        )

        self.section5_up = ENetUpsampleBottleneck(64, 16)
        self.section5_convs = ENetRegularBottleneck(16, 16)

        self.head = nn.ConvTranspose2d(
            16, num_classes, 3, stride=2, padding=1, bias=False
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        out: Tensor = x
        input_size = out.shape
        out = self.initial(out)

        section1_size = out.shape
        out, section1_indices = self.section1_down(out)
        out = self.section1_convs(out)

        section2_size = out.shape
        out, section2_indices = self.section2_down(out)
        out = self.section2_convs(out)

        out = self.section3_convs(out)

        out = self.section4_up(out, section2_indices, section2_size)
        out = self.section4_convs(out)

        out = self.section5_up(out, section1_indices, section1_size)
        out = self.section5_convs(out)

        out = self.head(out, output_size=input_size)

        return {"out": out}


class ENet_Weights(SegWeightsEnum):
    CITYSCAPES_FINE = SegWeights(
        "enet/enet-cityscapes-512x1024-20250304.pth",
        CITYSCAPES_LABELS,
        "Trained on Cityscapes (fine) dataset",
    )
    DEFAULT = CITYSCAPES_FINE


# not name it enet to prevent name clashing with module
@register_model("enet")
def enet_original(
    num_classes: int | None = None,
    weights: str | None = None,
    progress: bool = True,
) -> ENet:
    """Construct a ENet model

    Args:
        num_classes: number of output classes of the model (including the background).
        weights: Not supported yet. Must be None
        progress: If True, display the download progress.
    """
    weights_model = ENet_Weights.resolve(weights)
    weights_model, _, num_classes = _validate_weights_input(
        weights_model, None, num_classes
    )
    model = ENet(num_classes)

    if weights_model is not None:
        state_dict = load_state_dict_from_url(weights_model.url, progress=progress)
        model.load_state_dict(state_dict)
    return model
