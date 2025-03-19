from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.segmentation.lraspp import LRASPP, lraspp_mobilenet_v3_large

from ..datasets import CITYSCAPES_LABELS, VOC_LABELS
from .backbones import ResNetBackbone, replace_layer_name
from .model_registry import SegWeights, SegWeightsEnum, register_model
from .model_utils import _generate_docstring, _validate_weights_input

register_model()(lraspp_mobilenet_v3_large)


class LRASPP_ResNet18_Weights(SegWeightsEnum):
    CITYSCAPES = SegWeights(
        "lraspp/lraspp_resnet18-cityscapes-512x1024.pth",
        CITYSCAPES_LABELS,
        "Trained on Cityscapes (fine) dataset",
    )
    DEFAULT = CITYSCAPES


@_generate_docstring("Lite R-ASPP Network model with a ResNet-34 backbone")
@register_model()
def lraspp_resnet18(
    num_classes: int | None = None,
    weights: LRASPP_ResNet18_Weights | str | None = None,
    progress: bool = True,
    weights_backbone: ResNet18_Weights | str | None = ResNet18_Weights.DEFAULT,
) -> LRASPP:
    weights_model = LRASPP_ResNet18_Weights.resolve(weights)
    weights_model, weights_backbone, num_classes = _validate_weights_input(
        weights_model, weights_backbone, num_classes
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
