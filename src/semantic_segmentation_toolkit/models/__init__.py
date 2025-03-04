from .bisenet import BiSeNet, BiSeNet_ResNet18_Weights, bisenet_resnet18
from .deeplabv3 import deeplabv3_resnet34
from .enet import ENet, enet_original
from .fcn import FCN_ResNet34_Weights, fcn_mobilenet_v3_large, fcn_resnet34, fcn_vgg16
from .lraspp import lraspp_resnet18
from .model_registry import (
    MODEL_WEIGHTS,
    MODEL_ZOO,
    SegWeights,
    SegWeightsEnum,
    register_model,
)
from .pspnet import PSPNet, PSPNET_ResNet50_Weights, pspnet_resnet50
