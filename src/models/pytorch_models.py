import sys
from pathlib import Path

from torchvision.models import segmentation

sys.path.append(str((Path(__file__) / "..").resolve()))
from model_registry import register_model

# register builtin models
register_model()(segmentation.deeplabv3_mobilenet_v3_large)
register_model()(segmentation.deeplabv3_resnet101)
register_model()(segmentation.deeplabv3_resnet50)
register_model()(segmentation.lraspp_mobilenet_v3_large)


def __import():
    """Placeholder to import this module, to load the register. Do not use this function"""
    pass
