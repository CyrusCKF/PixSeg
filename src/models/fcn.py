import sys
from pathlib import Path

from torch import nn
from torch.hub import load_state_dict_from_url
from torchvision.models import *  # type: ignore
from torchvision.models import segmentation
from torchvision.models.segmentation.fcn import FCN, FCNHead

sys.path.append(str((Path(__file__) / "..").resolve()))
from backbones import *
from model_api import SegWeights, SegWeightsEnum
from model_registry import register_model

sys.path.append(str((Path(__file__) / "../../..").resolve()))
from src.datasets.pytorch_datasets import VOC_LABELS
from src.utils.transform import DataAugment

register_model()(segmentation.fcn_resnet50)
register_model()(segmentation.fcn_resnet101)


class FCN_ResNet34_Weights(SegWeightsEnum):
    VOC2012 = SegWeights(
        "https://github.com/CyrusCKF/segmentic-segmentation-toolkit/releases/download/fcn/fcn_resnet34-voc2012-500x500.pth",
        DataAugment,
        VOC_LABELS,
        "Trained on PASCAL VOC 2012 dataset",
    )
    DEFAULT = VOC2012


@register_model()
def fcn_resnet34(
    num_classes: int | None = None,
    weights: FCN_ResNet34_Weights | str | None = None,
    progress: bool = True,
    aux_loss: bool = False,
    weights_backbone: ResNet34_Weights | str | None = ResNet34_Weights.DEFAULT,
) -> nn.Module:
    weights_model = FCN_ResNet34_Weights.resolve(weights)
    if num_classes is None:
        num_classes = 21 if weights_model is None else len(weights_model.labels)
    if weights_model is not None:
        weights_backbone = None
        if num_classes != len(weights_model.labels):
            raise ValueError(
                f"Model weights {weights_model} expect number of classes"
                f"={len(weights_model.labels)}, but got {num_classes}"
            )

    backbone_model = resnet34(weights=weights_backbone, progress=progress)
    backbone = ResNetBackbone(backbone_model)
    replace_layer_name(backbone, {-1: "out", -2: "aux"})

    channels = backbone.layer_channels()
    aux_classifier = FCNHead(channels["aux"], num_classes) if aux_loss else None
    classifier = FCNHead(channels["out"], num_classes)
    model = FCN(backbone, classifier, aux_classifier)

    if weights_model is not None:
        state_dict = load_state_dict_from_url(weights_model.url, progress=progress)
        model.load_state_dict(state_dict)
    return model


@register_model()
def fcn_mobilenet_v3_large(
    num_classes: int | None = None,
    weights: str | None = None,
    progress: bool = True,
    aux_loss: bool = False,
    weights_backbone: (
        MobileNet_V3_Large_Weights | str | None
    ) = MobileNet_V3_Large_Weights.DEFAULT,
) -> nn.Module:
    if weights is not None:
        raise NotImplementedError("Weights is not supported yet")
    if num_classes is None:
        num_classes = 21

    backbone_model = mobilenet_v3_large(weights=weights_backbone, progress=progress)
    backbone = MobileNetV3Backbone(backbone_model)
    replace_layer_name(backbone, {-1: "out", -4: "aux"})

    channels = backbone.layer_channels()
    aux_classifier = FCNHead(channels["aux"], num_classes) if aux_loss else None
    classifier = FCNHead(channels["out"], num_classes)
    return FCN(backbone, classifier, aux_classifier)


@register_model()
def fcn_vgg16(
    num_classes: int | None = None,
    weights: str | None = None,
    progress: bool = True,
    aux_loss: bool = False,
    weights_backbone: VGG16_Weights | str | None = VGG16_Weights.DEFAULT,
) -> nn.Module:
    if weights is not None:
        raise NotImplementedError("Weights is not supported yet")
    if num_classes is None:
        num_classes = 21

    backbone_model = vgg16(weights=weights_backbone, progress=progress)
    backbone = VGGBackbone(backbone_model)
    replace_layer_name(backbone, {-1: "out", -2: "aux"})

    channels = backbone.layer_channels()
    aux_classifier = FCNHead(channels["aux"], num_classes) if aux_loss else None
    classifier = FCNHead(channels["out"], num_classes)
    return FCN(backbone, classifier, aux_classifier)
