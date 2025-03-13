from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.segmentation.lraspp import LRASPP, lraspp_mobilenet_v3_large

from .backbones import ResNetBackbone, replace_layer_name
from .model_registry import register_model
from .model_utils import _generate_docstring, _validate_weights_input

register_model()(lraspp_mobilenet_v3_large)


@_generate_docstring("Lite R-ASPP Network model with a ResNet-34 backbone")
@register_model()
def lraspp_resnet18(
    num_classes: int | None = None,
    weights: str | None = None,
    progress: bool = True,
    weights_backbone: ResNet18_Weights | str | None = ResNet18_Weights.DEFAULT,
) -> LRASPP:
    if weights is not None:
        raise NotImplementedError("Weights is not supported yet")
    weights_model, weights_backbone, num_classes = _validate_weights_input(
        None, weights_backbone, num_classes
    )

    backbone_model = resnet18(weights=weights_backbone, progress=progress)
    backbone = ResNetBackbone(backbone_model)
    replace_layer_name(backbone, {-1: "high", -2: "low"})

    channels = backbone.layer_channels()
    model = LRASPP(backbone, channels["low"], channels["high"], num_classes)

    if weights_model is not None:
        state_dict = load_state_dict_from_url(weights_model.url, progress=progress)
        model.load_state_dict(state_dict)
    return model
