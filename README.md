# PixSeg

Pixel segmentation (a.k.a. semantic segmentation) is a task where the goal is to classify each pixel in an image into a class. This project is a lightweight yet fully featured and easy-to-use package for such task. It contains PyTorch implementation of various deep learning components, including models, pertrained weights, datasets, loss functions, etc.

## Highlights

- **Plug and play** - Custom models, backbones, datasets and other components with interface similar to *PyTorch* counterparts
- **Lightweight** - Only dependency is pytorch (torch & torchvision) for standard versions
- **Fully featured** - Provide visualization, metrics and data transform tools specifically for semantic segmentation
- **Extensible** - Easily extend the registered components to include in your training config
- **Comprehensive logging** - All hyperparameters and results are logged for multiple providers (local, WandB, Tensorboard, etc) using the custom trainer and config

## Kickoff

There **3 ways** to start using this project

### *Option 1* With `torch.hub`

This allows you to use model and pretrained weights **without installation** (you still need PyTorch). See [*here*](#models) for all models and their weights.

```python
import torch
print(torch.hub.list("CyrusCKF/PixSeg", force_reload=True, trust_repo=True))
print(torch.hub.help("CyrusCKF/PixSeg", "bisenet_resnet18"))
model = torch.hub.load("CyrusCKF/PixSeg", "bisenet_resnet18", weights="DEFAULT")
```

Refer to [doc/using_hub.ipynb](doc/using_hub.ipynb) for more usage. As this project shares the same interface as TorchVision, you may also refer to [the official doc](<https://pytorch.org/docs/main/hub.html>) or this section [Using models from Hub](https://pytorch.org/vision/main/models.html#using-models-from-hub) on how to use vision models via PyTorch Hub.

### *Option 2* Import as package

Run `pip install pixseg`  
This supports models, datasets, loss functions, etc and some utility modules. See [*here*](#usage-examples) for usage.

Optionally, to use config and trainer, run `pip install pixseg[full]`  

### *Option 3* Clone this project

1. Run `git clone https://github.com/CyrusCKF/PixSeg.git`
2. Create environment and install PyTorch <https://pytorch.org/get-started/locally/>
3. Run `pip install -e .[dev]`
4. See [*here*](#usage-examples) for usage.

## Usage examples

Here shows examples when you clone this project (*Option 3*). But the same concept applies for installing as package (*Option 2*)

### Dataset and its info

```python
from pathlib import Path
from torch.utils.data import DataLoader
from src.pixseg.datasets import ADE20K, resolve_metadata
from src.pixseg.utils.transform import SegmentationTransform

root = Path(r"path/to/ADE20K")
# Preset metadata of ADE20K, which includes number of classes, labels, index for background, etc
metadata = resolve_metadata("ADE20K")
# Make sure the data is of the right format for the model input
transforms = SegmentationTransform(size=(512, 512), mask_fill=metadata.ignore_index)
dataset = ADE20K(root, split="training", transforms=transforms)
train_loader = DataLoader(dataset, batch_size=4, drop_last=True, shuffle=True)
```

### Network components

```python
from torch.optim.lr_scheduler import PolynomialLR
from src.pixseg.learn import DiceLoss, Padam
from src.pixseg.models import PSPNET_ResNet50_Weights, pspnet_resnet50
from src.pixseg.utils.transform import SegmentationAugment

# Create PSPNet with ResNet-50 backbone and activate auxillary loss
model = pspnet_resnet50(num_classes=metadata.num_classes, aux_loss=True)
# Or initialize with pretrained weights
model = pspnet_resnet50(weights=PSPNET_ResNet50_Weights.DEFAULT)
criterion = DiceLoss(ignore_index=metadata.ignore_index)
optimizer = Padam(model.parameters(), lr=0.1, weight_decay=5e-4, partial=0.125)
lr_scheduler = PolynomialLR(optimizer, total_iters=100, power=0.9)
```

### Training loop and utils

Utils are mostly supplementary functions to training loops. Here is a brief example of using them.

```python
from torch import Tensor
from src.pixseg.utils.metrics import MetricStore
from src.pixseg.utils.rng import seed
from src.pixseg.utils.visual import exhibit_figure, plot_confusion_matrix

# Fix random seeds of random, numpy and pytorch
seed(42)
# This project separates data transforms and data augmentations, so that visualizations can show the original images
train_augment = SegmentationAugment(hflip=0.5, mask_fill=metadata.ignore_index)
for i in range(100):  # Set your number of epochs
    model.train()
    # Store and calculate metrics efficiently
    train_ms = MetricStore(metadata.num_classes)
    for j, (images, masks) in enumerate(train_loader):
        images, masks = train_augment(images, masks)
        ...
        logits: dict[str, Tensor] = model(images)
        train_ms.store_results(masks, logits["out"].argmax(1))
        ...

    # Print training results
    print(train_ms.summarize())
    # Visualize confusion matrix and save as image
    plot_confusion_matrix(train_ms.confusion_matrix, metadata.labels)
    exhibit_figure(show=False, save_to=Path("confusion_matrix.png"))
```

### More on usage

For general information about using models and pretrained weights, you may refere to <https://pytorch.org/vision/main/models.html#general-information-on-pre-trained-weights>, which shares the same interface. This section [Using models from Hub](https://pytorch.org/vision/main/models.html#using-models-from-hub) shows how to use models via PyTorch Hub.

You may refer to [tasks/minimal_training.ipynb](tasks/minimal_training.ipynb) to see how to implement a simple training and evaluation loop. Pretrained weights and their training details can be found in the [*Release*](https://github.com/CyrusCKF/PixSeg/releases) side bar. All of them are also exposed through enums in code.

If you installed the **full** version or cloned this project, you can go to [tasks/training.ipynb](tasks/training.ipynb) to see how to start training with config and loggers. You may check [doc/config_doc.ipynb](doc/config_doc.ipynb) to customize the config. In [tasks/inference.ipynb](tasks/inference.ipynb) also demonstrates a lot of functions to improve performance in test time

## Features

Some of them is available in PyTorch. They are listed here for clarity and to show that the custom components are fully compliant with them.

### Datasets

Each dataset returns a tuple of (image, mask)

#### General Scene

- **ADE20K** | [website](https://ade20k.csail.mit.edu/) • [download](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip) • [benchmark](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k)
- **COCO-Stuff** | [website](https://github.com/nightrome/cocostuff) • [benchmark](https://paperswithcode.com/sota/semantic-segmentation-on-coco-stuff-test)
- **Semantic Boundaries Dataset** | [download](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz)
- **PASCAL VOC** | [website](http://host.robots.ox.ac.uk/pascal/VOC/) • [benchmark](https://paperswithcode.com/sota/semantic-segmentation-on-pascal-voc-2012)

#### Street scene

- **BDD100K** Using the *10K Images* and *Segmentation* (The official doc is not accessible now?) | [website](http://bdd-data.berkeley.edu/) • [benchmark](https://paperswithcode.com/sota/semantic-segmentation-on-bdd100k-val)
- **Cityscapes** | [website](https://www.cityscapes-dataset.com/) • [benchmark](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes-val)
- **Mapillary Vistas** Using version 2.0 | [website](https://www.cityscapes-dataset.com/) • [benchmark](https://paperswithcode.com/sota/semantic-segmentation-on-mapillary-val)

#### Human parts

- **Look Into Person** | [website](https://sysu-hcp.net/lip/) • [benchmark](https://paperswithcode.com/sota/semantic-segmentation-on-lip-val)

### Models

Each model returns a dict like `{ "out": Tensor }`. May contain other keys like *"aux"* for auxillary logits or [Deeply-Supervised Nets](https://arxiv.org/abs/1409.5185).

- **BiSeNet** BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation (2018) | [paper](https://arxiv.org/abs/1808.00897) • [weights](https://github.com/CyrusCKF/PixSeg/releases/tag/bisenet)
- **DeepLabv3** Rethinking Atrous Convolution for Semantic Image Segmentation (2017) | [paper](https://arxiv.org/abs/1706.05587) • [weights](https://github.com/CyrusCKF/PixSeg/releases/tag/deeplabv3)
- **ENet** ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation (2016) | [paper](https://arxiv.org/abs/1606.02147) • [weights](https://github.com/CyrusCKF/PixSeg/releases/tag/enet)
- **FCN** Fully Convolutional Networks for Semantic Segmentation (2014) | [paper](https://arxiv.org/abs/1411.4038) • [weights](https://github.com/CyrusCKF/PixSeg/releases/tag/fcn)
- **LRASPP** Searching for MobileNetV3 (2019) | [paper](https://arxiv.org/abs/1905.02244) • weights TBD
- **PSPNet** Pyramid Scene Parsing Network (2016) | [paper](https://arxiv.org/abs/1612.01105) • [weights](https://github.com/CyrusCKF/PixSeg/releases/tag/pspnet)
- **SFNet-Lite** (2023) | [reference](https://github.com/lxtGH/SFSegNets) • weights TBD
- **SFNet** Semantic Flow for Fast and Accurate Scene Parsing (2020) | [paper](https://arxiv.org/abs/2002.10120) • [weights](https://github.com/CyrusCKF/PixSeg/releases/tag/sfnet)
- **UPerNet** Unified Perceptual Parsing for Scene Understanding (2018) | [paper](https://arxiv.org/abs/1807.10221) • weights TBD

### Model weights

Available in <https://github.com/CyrusCKF/PixSeg/releases>. The model files can be found inside the Assets tab. The specification of each pretrained model is shown.

In particular, *mIoU(tta)* means evaluating with these test-time augmentations: multi-scale of (0.5, 0.75, 1.0, 1.25, 1.5) x horizontal flip of both on and off. *MACs* is the number of multiply-accumulate operations. *Memory* and *MACs* are calculated with batch size of 1. *Time* is estimated using eval mode with batch size of 1 on RTX3070.

### Backbones

Each backbone returns an ordered dict of features, from fine to coarse.

- **MobileNetV3** Searching for MobileNetV3 (2019) | [paper](https://arxiv.org/abs/1905.02244)
- **ResNet** Deep Residual Learning for Image Recognition (2015) | [paper](https://arxiv.org/abs/1512.03385)
- **VGG** Very Deep Convolutional Networks for Large-Scale Image Recognition (2014) | [paper](https://arxiv.org/abs/1409.1556)
- **Xception** Xception: Deep Learning with Depthwise Separable Convolutions (2016) | [paper](https://arxiv.org/abs/1610.02357) • [weights](https://github.com/CyrusCKF/PixSeg/releases/tag/xception)

### Loss functions

- **Cross Entropy Loss**
- **Dice Loss** Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations | [paper](https://arxiv.org/abs/1707.03237v3)
- **Focal Loss** Focal Loss for Dense Object Detection | [paper](https://arxiv.org/abs/1708.02002v2)

### Loss weights

Each weighting takes a dataset and calculates the loss weights for each class.

Supported algorithms: **Inverse frequency** | **Inverse square root frequency** | **Inverse log frequency** | **Inverse effective number of samples** [(reference)](https://arxiv.org/abs/1901.05555v1)

### Optimizers

- **SGD**
- **Adam** Adam: A Method for Stochastic Optimization | [paper](https://arxiv.org/abs/1412.6980)
- **Padam** Closing the Generalization Gap of Adaptive Gradient Methods in Training Deep Neural Networks | [paper](https://arxiv.org/abs/1806.06763)

### LR schedulers

Supported all algorithms provided by PyTorch, including: **StepLR** | **PolynomialLR** | **OneCycleLR** | **CosineAnnealingLR**

### Transforms

This project separates data transforms and data augmentations, so that visualizations can show the original images

- Data transforms

  Ensure images and targets are in the correct type and shape. See [source code](https://github.com/CyrusCKF/PixSeg/blob/main/src/pixseg/utils/transform.py#L132) of `SegmentationTransform` for details

- Data augmentations

  Provide a parametrised interface of common augmentations. See [source code](https://github.com/CyrusCKF/PixSeg/blob/main/src/pixseg/utils/transform.py#L164) of `SegmentationAugment` for details

### Metrics

Let $n$ be number of classes and $S_i$ be number of ground truths in class $i$, i.e. $S_i = TP_i + FN_i$. Denote total number of samples to be $S = \sum_{i=1}^n S_i = TP + TN + FP + FN$

```math
\begin{flalign}
\text{accuracy} &= \frac{TP}{S} && \newline
\text{mean accuracy} &= \frac{1}{n} \times \sum_{i=1}^n \frac{TP_i}{S_i} && \newline
\text{mean IoU} &= \frac{1}{n} \times \sum_{i=1}^n \frac{TP_i}{TP_i + FP_i + FN_i} && \newline
\text{frequency-weighted IoU} &= \frac{1}{S} \times \sum_{i=1}^n  S_i \times \frac{TP_i}{TP_i + TN_i + FP_i} && \newline
\text{Dice} &= \frac{1}{n} \times \sum_{i=1}^n \frac{2 \times TP_i}{2 \times TP_i + FP_i + FN_i} && \newline
\end{flalign}
```

### Loggers\*

Each logger contains different implmentations of hook to record results. See [Training outputs](#training-outputs) for the behaviour of each of them. All cexcept *Local* can be toggled off.

### Trainer\*

Combine components in each part to form a complete training loop, with evaluation after each epoch.

### Config\*

See [doc\sample_config.toml](doc\sample_config.toml) for config sample.  
See [doc\config_doc.ipynb](doc\config_doc.ipynb) for explanation of each field.

### Test time\*

Test-time augmentations and sliding window inference. Other techniques don't really do much. See [tasks\inference.ipynb](tasks\inference.ipynb) for demonstrations.  

\* Only available in the full version

## Training outputs

In the **\[full\]** version, trainer can record results locally or to various service providers. This previews the bahaviour of each logger. You may also implement your own.

### Local

Logs, confusion matrix, snapshots, checkpoints and models

<p float="left">
<img src="doc\assets\cm_example.png" alt="confusion matrix example" width="30%"/>
<img src="doc\assets\snapshot_example.png" alt="snapshot example" width="30%"/>
</p>

### WandB

Most params in config will uploaded. Metrics are tracked each epoch.

<img src="doc\assets\wandb_table.png" alt="wandb table" width="70%"/>
<img src="doc\assets\wandb_graph.png" alt="wandb graph" width="70%"/>

### Tensorboard

TODO

## Plans

- More models (types, backbones)
  - Planned: PIDNet, Deeplabv3+, RegSeg, EfficientNet
- More datasets
  - Planned: [LoveDA](https://github.com/Junjue-Wang/LoveDA), [iSAID](https://captain-whu.github.io/iSAID/), [ISPRS](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx)
  - Unify dataset of similar nature, e.g. dash cam, aerial images
- More pretrained weights
  - At least one weights for each model builder
  - At least one pretrained lightweight model and performant model for each dataset
- More loggers
- Proofread README, docs and docstring
