dependencies = ["torch"]
from src.semantic_segmentation_toolkit.models.bisenet import (
    bisenet_resnet18,
    bisenet_resnet50,
    bisenet_xception,
)
from src.semantic_segmentation_toolkit.models.deeplabv3 import deeplabv3_resnet18
from src.semantic_segmentation_toolkit.models.enet import enet_original
from src.semantic_segmentation_toolkit.models.fcn import fcn_vgg16
from src.semantic_segmentation_toolkit.models.lraspp import lraspp_resnet18
from src.semantic_segmentation_toolkit.models.model_registry import (
    get_model,
    get_model_builder,
    get_model_weights,
    get_weight,
    list_models,
)
from src.semantic_segmentation_toolkit.models.pspnet import pspnet_resnet50
from src.semantic_segmentation_toolkit.models.sfnet import (
    sfnet_resnet18,
    sfnet_resnet101,
)
from src.semantic_segmentation_toolkit.models.sfnet_lite import (
    sfnet_lite_resnet18,
    sfnet_lite_resnet101,
)
from src.semantic_segmentation_toolkit.models.upernet import (
    upernet_resnet18,
    upernet_resnet101,
)
