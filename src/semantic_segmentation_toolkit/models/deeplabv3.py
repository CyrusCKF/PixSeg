from torch import nn
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.segmentation.deeplabv3 import (
    DeepLabHead,
    DeepLabV3,
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet50,
    deeplabv3_resnet101,
)
from torchvision.models.segmentation.fcn import FCNHead

from .backbones import ResNetBackbone, replace_layer_name
from .model_registry import register_model
from .model_utils import _generate_docstring, _validate_weights_input

register_model()(deeplabv3_mobilenet_v3_large)
register_model()(deeplabv3_resnet50)
register_model()(deeplabv3_resnet101)


@_generate_docstring("DeepLabV3 model with a ResNet-18 backbone")
@register_model()
def deeplabv3_resnet18(
    num_classes: int | None = None,
    weights: str | None = None,
    progress: bool = True,
    aux_loss: bool = False,
    weights_backbone: ResNet18_Weights | str | None = ResNet18_Weights.DEFAULT,
) -> DeepLabV3:
    if weights is not None:
        raise NotImplementedError("Weights is not supported yet")
    weights_model, weights_backbone, num_classes = _validate_weights_input(
        None, weights_backbone, num_classes
    )

    backbone_model = resnet18(weights=weights_backbone, progress=progress)
    backbone = ResNetBackbone(backbone_model)
    replace_layer_name(backbone, {-1: "out", -2: "aux"})

    channels = backbone.layer_channels()
    classifier = DeepLabHead(channels["out"], num_classes)
    aux_classifier = FCNHead(channels["aux"], num_classes) if aux_loss else None
    model = DeepLabV3(backbone, classifier, aux_classifier)

    if weights_model is not None:
        state_dict = load_state_dict_from_url(weights_model.url, progress=progress)
        model.load_state_dict(state_dict)
    return model
