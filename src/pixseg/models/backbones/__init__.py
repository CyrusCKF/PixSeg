"""Each backbone module returns an ordered dict of feature maps.

Usually, it goes from shallow, fine and low-level feature to deep, coarse and high-level.

Most backbone should have pretrained weights; otherwise it is too impractical to train
the whole segmentation model.
"""

from .backbone_utils import replace_layer_name
from .mobilenet_v3 import MobileNetV3Backbone
from .resnet import ResNetBackbone
from .vgg import VGGBackbone
from .xception import Xception, Xception_Weights, XceptionBackbone, xception_original
