import sys
from pathlib import Path

from torch import nn
from torchvision.models import resnet, segmentation
from torchvision.models.resnet import ResNet34_Weights
from torchvision.models.segmentation import fcn

sys.path.append(str((Path(__file__) / "..").resolve()))
from backbones import ResNetBackbone, replace_getter_layer
from model_registry import register_model

register_model()(segmentation.fcn_resnet50)
register_model()(segmentation.fcn_resnet101)


@register_model()
def fcn_resnet34(
    num_classes: int | None = None,
    weights: str | None = None,
    progress: bool = True,
    aux_loss: bool = False,
    weights_backbone: ResNet34_Weights | str | None = ResNet34_Weights.IMAGENET1K_V1,
) -> nn.Module:
    if weights is not None:
        raise NotImplementedError("Weights is not supported yet")
    if num_classes is None:
        num_classes = 21

    backbone_model = resnet.resnet34(weights=weights_backbone, progress=progress)
    backbone = ResNetBackbone(backbone_model)
    replace_getter_layer(backbone, True, aux_loss)

    channels = backbone.layer_channels()
    aux_classifier = fcn.FCNHead(channels["aux"], num_classes) if aux_loss else None
    classifier = fcn.FCNHead(channels["out"], num_classes)
    return fcn.FCN(backbone, classifier, aux_classifier)
