# Semantic Segmentation Toolkit

This project is a lightweight yet fully featured and easy-to-use package for semantic segmentation. It contains PyTorch implementation of various components in semantic segmentation, including models, datasets, loss functions, etc.

## Highlights

- **Plug and play** - Custom models, backbones, datasets and other components with interface similar to *PyTorch* counterparts
- **Lightweight** - Only dependency is pytorch (torch & torchvision) for standard versions
- **Fully featured** - Provide visualization, metrics and data transform tools specifically for semantic segmentation
- **Extensible** - Easily extend the registered components to include in your training config
- **Comprehensive logging** - All hyperparameters and results are logged for multiple providers (local, WandB, Tensorboard, etc) using the custom trainer and config

## Kickoff

There **3 ways** to start using this project

### *Option 1* With `torch.hub`

This allows you to use model and pretrained weights **without installation** (you still need pytorch). See the [*here*](#models) for all models and their weights.

```python
import torch
print(torch.hub.list("CyrusCKF/semantic-segmentation-toolkit", trust_repo="True"))
model = torch.hub.load(
    "CyrusCKF/semantic-segmentation-toolkit",
    "fcn_resnet34",
    weights="DEFAULT", # Load pretrained weights
)
```

Refer to <https://pytorch.org/docs/main/hub.html> for more usage

### *Option 2* Import as package

Run `pip install semantic-segmentation-toolkit`  
This supports models, datasets, loss functions, etc and some utility modules. See [*here*](#usage-examples) for usage.

Optionally, to use config and trainer, run `pip install semantic-segmentation-toolkit[full]`  

### *Option 3* Clone this project

1. Run `git clone https://github.com/CyrusCKF/semantic-segmentation-toolkit.git`
2. Create environment and install PyTorch
3. Run `pip install -e .[dev]`
4. See [*here*](#usage-examples) for usage.

## Usage examples

This shows examples when you install this project as package (*Option 2*). But the same concept applies for *Option 3*

### Dataset

```python
from pathlib import Path
from torch.utils.data import DataLoader
from semantic_segmentation_toolkit.datasets import CityscapesClass, resolve_metadata
from semantic_segmentation_toolkit.utils.transform import SegmentationAugment, SegmentationTransform

root = Path(r"..\Cityscapes")
# Preset metadata of Cityscapes
metadata = resolve_metadata("Cityscapes") 
# Make sure the data is of the right format for the model input
transforms = SegmentationTransform(size=(512, 1024), mask_fill=metadata.ignore_index)
# CityscapesClass focuses only on the trainIDs of Cityscapes classes
dataset = CityscapesClass(root=root, split="train", target_type="semantic", transforms=transforms)
data_loader = DataLoader(dataset, batch_size=4, drop_last=True, shuffle=True)
```

### Network

```python
from semantic_segmentation_toolkit.models import fcn_resnet34, FCN_ResNet34_Weights
from semantic_segmentation_toolkit.learn import DiceLoss, Padam

# Create FCN with ResNet34 backbone and activate auxillary loss
model = fcn_resnet34(num_classes=metadata.num_classes, aux_loss=True)
# Or initialize with pretrained weights
model = fcn_resnet34(weights=FCN_ResNet34_Weights.VOC2012)
criterion = DiceLoss(ignore_index=metadata.ignore_index)
optimizer = Padam(model.parameters(), lr=0.1, weight_decay=5e-4, partial=0.125)
```

### Utils

Utils are mostly supplementary functions to training loops. Here is a brief example of using them.

```python
from semantic_segmentation_toolkit.utils.metrics import MetricStore
from semantic_segmentation_toolkit.utils.rng import seed
from semantic_segmentation_toolkit.utils.visual import plot_confusion_matrix, exhibit_figure

# Fix random seeds of random, numpy and pytorch
seed(42) 
# This project separates data transforms and data augmentations, so that visualizations can show the original images
train_augment = SegmentationAugment(hflip=0.5, mask_fill=metadata.ignore_index)
for i in range(num_epochs):
    model.train()
    # Store and calculate metrics efficiently
    train_ms = MetricStore(metadata.num_classes)
    for j, (images, masks) in enumerate(train_loader):
        images, masks = train_augment(images, masks)
        ...
        logits: dict[str, Tensor] = model(images)
        train_ms.store_results(masks, logits["out"].argmax(1))
        ...

    # Log training results
    logging.info(train_ms.summarize())
    # Visualize confusion matrix and save as image
    plot_confusion_matrix(train_ms.confusion_matrix, metadata.labels)
    exhibit_figure(show=False, save_to=Path("confusion_matrix.png"))
```

### More on usage

You may refer to [tasks\minimal_training.ipynb](tasks\minimal_training.ipynb) to see how to implement a simple training and evaluation loop. Pretrained weights and their training details can be found in the [*Release*](https://github.com/CyrusCKF/semantic-segmentation-toolkit/releases) side bar. All of them are also exposed through enums in code.

If you installed the **full** version or cloned this project, you can go to [tasks\training.ipynb](tasks\training.ipynb) to see how to start training with config and loggers. You may check [doc\config_doc.ipynb](doc\config_doc.ipynb) to customize the config. In [tasks\inference.ipynb](tasks\inference.ipynb) also demonstrates a lot of functions for improve performance in test time

## Training outputs

In the **\[full\]** version, trainer can record results locally or to various service providers. This demonstrates the default bahaviour of each loggers. You may also implement your own.

### Local

- Logs
- Checkpoints and models
- Confusion matrix
- Snapshots

### WandB

### Tensorboard

## Features

Some of them is available in PyTorch. They are listed here for clarity and to show that the custom components are fully compliant with them.

### Models

Each model returns a dict like `{ "out": Tensor }`. May contain *"aux"* for auxillary logits.

**FCN** Fully Convolutional Networks for Semantic Segmentation (2014) | [paper](https://arxiv.org/abs/1411.4038) • [code](src/semantic_segmentation_toolkit/models/fcn.py) • [weights](https://github.com/CyrusCKF/semantic-segmentation-toolkit/releases/tag/fcn)

**PSPNet** Pyramid Scene Parsing Network (2016) | [paper](https://arxiv.org/abs/1612.01105) • [code](src/semantic_segmentation_toolkit/models/pspnet.py) • [weights](https://github.com/CyrusCKF/semantic-segmentation-toolkit/releases/tag/pspnet)

**DeepLabv3** Rethinking Atrous Convolution for Semantic Image Segmentation (2017) | [paper](https://arxiv.org/abs/1706.05587) • [code](src/semantic_segmentation_toolkit/models/pytorch_models.py) • [weights](https://github.com/CyrusCKF/semantic-segmentation-toolkit/releases/tag/deeplabv3)

### Backbone

Each backbone returns an ordered dict of features, from fine to coarse.

**VGG** Very Deep Convolutional Networks for Large-Scale Image Recognition (2014) | [paper](Very Deep Convolutional Networks for Large-Scale Image Recognition) • [code](src/semantic_segmentation_toolkit/models/backbones/vgg.py)

### Datasets

Each dataset returns a tuple of (image, mask)

VOC | [website]() • [code]() • [paperswithcode]()

### Loss functions

DiceLoss | [code]()

#### Loss weights

Each weighting takes dataset and calculate the loss weights for each class.

Effective number of samples | [paper]() • [code]()

### Optimizers

Padam ([paper]() • [code]())

### LR schedulers

All are PyTorch builtin and registered to be used in config.

### Transforms

This project separates data transforms and data augmentations, so that visualizations can show the original images

- Data transforms

- Data augmentations

### Metrics

Accuracy $`= (1/n) * \sum TP / (TP + TN + FP + FN)`$

### Loggers\*

Each logger contains different implmentations of hook to record results. See [Training outputs](#training-outputs) for the behaviour of each of them. All cexcept *Local* can be toggled off.

### Trainer\*

Combine components in each part to form a complete training loop, with evaluation after each epoch.

### Config\*

See [doc\sample_config.toml](doc\sample_config.toml) for config sample.  
See [doc\config_doc.ipynb](doc\config_doc.ipynb) for explanation of each field.

### Test time\*

Test-time augmentations and sliding window inference. Others don't really do much

\* Only available in the full version

## Plans

- More models (types, backbones, builders)
- More pretrained weights
- More datasets
- Proofread this README
