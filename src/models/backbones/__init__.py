"""Each backbone module returns an ordered dict of feature maps.

Usually, it goes from shallow, fine and low-level feature to deep, coarse and high-level.

Most backbone should have pretrained weights; otherwise it is too impractical to train
the whole segmentation model.
"""

from .resnet import ResNetBackbone
