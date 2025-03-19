from torch.hub import load_state_dict_from_url
from torchvision.models import VGG16_Weights, vgg16
from torchvision.models.segmentation.fcn import (
    FCN,
    FCNHead,
    fcn_resnet50,
    fcn_resnet101,
)

from ..datasets import VOC_LABELS
from .backbones import *
from .model_registry import SegWeights, SegWeightsEnum, register_model
from .model_utils import _generate_docstring, _validate_weights_input

register_model()(fcn_resnet50)
register_model()(fcn_resnet101)


@_generate_docstring("Fully-Convolutional Network model with a VGG-16 backbone.")
@register_model()
def fcn_vgg16(
    num_classes: int | None = None,
    weights: str | None = None,
    progress: bool = True,
    aux_loss: bool = False,
    weights_backbone: VGG16_Weights | str | None = VGG16_Weights.DEFAULT,
) -> FCN:
    if weights is not None:
        raise NotImplementedError("Weights is not supported yet")
    _, weights_backbone, num_classes = _validate_weights_input(
        None, weights_backbone, num_classes
    )

    backbone_model = vgg16(weights=weights_backbone, progress=progress)
    backbone = VGGBackbone(backbone_model)
    replace_layer_name(backbone, {-1: "out", -2: "aux"})

    channels = backbone.layer_channels()
    classifier = FCNHead(channels["out"], num_classes)
    aux_classifier = FCNHead(channels["aux"], num_classes) if aux_loss else None
    return FCN(backbone, classifier, aux_classifier)
