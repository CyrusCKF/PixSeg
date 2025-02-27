# segmentic-segmentation-toolkit

A simple semantic segmentation package using PyTorch

## Main features

- **Plug and play** - Custom models, backbones, datasets and other components with interface simmilar to *pytorch*
- **Lightweight** - only dependencies are pytorch (torch & torchvision) for standard versions
- **Fully featured** - provide visualization, metrics and data transform specifically for semantic segmentation
- **Extensible** - you can easily extend the registered components to include in your training config
- **Comprehensive logging** All hyperparameters and results are logged for multiple providers (local, WandB, Tensorboard, etc) using the custom trainer and config

## Using this project

There **3 ways** to use this project

### *Option 1* With `torch.hub`

This allows you to use model and pretrained weights **without installation** (you still need pytorch)

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
This supports models, datasets, criterions, optimizers, etc and some utility modules.

Optionally, to use config and trainer, run `pip install semantic-segmentation-toolkit[full]`  

### *Option 3* Clone this project

1. Run `git clone https://github.com/CyrusCKF/semantic-segmentation-toolkit.git`
2. Create environment and install PyTorch
3. Run `pip install -e .[dev]`

## Overview

You may check [doc\config_doc.ipynb](doc\config_doc.ipynb) for all available models, datasets, and other components.

You may refer to [tasks\minimal_training.ipynb](tasks\minimal_training.ipynb) to see how to implement a simple training and evaluation loop.

If you installed the full version or cloned this project, you can go to [tasks\training.ipynb](tasks\training.ipynb) to see how to start training with config and loggers. In [tasks\inference.ipynb](tasks\inference.ipynb) also demonstrates a lot of functions for imprrve performance in test time
