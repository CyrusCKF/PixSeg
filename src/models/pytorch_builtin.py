import sys
from pathlib import Path

from torchvision.models import segmentation

sys.path.append(str((Path(__file__) / "..").resolve()))
from registry import register_model

# register builtin models
register_model()(segmentation.deeplabv3_mobilenet_v3_large)
register_model()(segmentation.deeplabv3_resnet101)
register_model()(segmentation.deeplabv3_resnet50)
register_model()(segmentation.fcn_resnet50)
register_model()(segmentation.fcn_resnet101)
register_model()(segmentation.lraspp_mobilenet_v3_large)


def _test():
    import torchinfo
    from registry import MODEL_ZOO

    print(MODEL_ZOO)
    builder = MODEL_ZOO["deeplabv3_mobilenet_v3_large"]
    model = builder()
    torchinfo.summary(model, (4, 3, 224, 512))


if __name__ == "__main__":
    _test()
