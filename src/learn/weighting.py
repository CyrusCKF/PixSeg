import sys
from pathlib import Path

import torch
import tqdm
from torch.utils.data import Dataset
from torchvision.transforms.v2 import functional as TF

sys.path.append(str((Path(__file__) / "..").resolve()))
from learning_zoo import CLASS_WEIGHTINGS, WeightingFunc


def register_weighting(name: str | None = None):
    def wrapper(func: WeightingFunc) -> WeightingFunc:
        key = func.__name__ if name is None else name
        if key in CLASS_WEIGHTINGS:
            raise ValueError(f"An entry is already registered under the name '{key}'.")
        CLASS_WEIGHTINGS[key] = func
        return func

    return wrapper


def count_classes(dataset: Dataset, num_classes: int):
    """Count the frequency of each class"""
    class_counts = torch.zeros([num_classes], dtype=torch.long)
    size = len(dataset)  # type: ignore
    for _, mask in tqdm.tqdm(iter(dataset), total=size, desc="count_classes"):
        mask_tensor = TF.to_image(mask)
        mask_counts = torch.bincount(mask_tensor.flatten(), minlength=num_classes)
        class_counts += mask_counts[:num_classes]
    return class_counts


@register_weighting()
def none(*args):
    return None


@register_weighting()
def class_frequency(dataset: Dataset, num_classes: int):
    class_freq = count_classes(dataset, num_classes)
    return class_freq.median() / class_freq.float()


@register_weighting()
def sqrt_frequency(dataset: Dataset, num_classes: int):
    class_freq = count_classes(dataset, num_classes)
    return class_freq.median().sqrt() / class_freq.sqrt()


@register_weighting()
def log_frequency(dataset: Dataset, num_classes: int):
    class_freq = count_classes(dataset, num_classes)
    return class_freq.median().log() / class_freq.log()


@register_weighting()
def effective_number(dataset: Dataset, num_classes: int):
    """Custom implementation of Effective Number of Samples

    Source: https://arxiv.org/pdf/1901.05555v1

    Beta is chosen so that the class weights are meaningful in the
    context of semantic segmentation
    """
    class_freq = count_classes(dataset, num_classes)
    beta = 1 - 1 / class_freq.median()
    effective_number = (1 - torch.pow(beta, class_freq)) / (1 - beta)
    return effective_number.median() / effective_number
