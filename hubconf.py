dependencies = ["torch"]
from src.pixseg.models.bisenet import (
    bisenet_resnet18,
    bisenet_resnet50,
    bisenet_xception,
)
from src.pixseg.models.deeplabv3 import deeplabv3_resnet18
from src.pixseg.models.enet import enet_original
from src.pixseg.models.fcn import fcn_vgg16
from src.pixseg.models.lraspp import lraspp_resnet18
from src.pixseg.models.model_registry import (
    get_model,
    get_model_builder,
    get_model_weights,
    get_weight,
    list_models,
)
from src.pixseg.models.pspnet import pspnet_resnet50
from src.pixseg.models.sfnet import sfnet_resnet18, sfnet_resnet101
from src.pixseg.models.sfnet_lite import sfnet_lite_resnet18, sfnet_lite_resnet101
from src.pixseg.models.upernet import upernet_resnet18, upernet_resnet101
