"""Contain collection of strategies used to enhance inference results during test time.

See `tasks/inference.ipynb` for demo and usage
"""

import itertools
from typing import Literal, Sequence

import numpy as np
import torch
from pydensecrf.utils import unary_from_softmax
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.transforms.v2 import functional as TF


def refine_prob_by_crf(prob: np.ndarray, image: Tensor | None, iter=5) -> np.ndarray:
    """Apply crf on softmax class-probabilities

    Reference https://github.com/lucasb-eyer/pydensecrf/blob/master/README.md

    Args:
        prob (float array (num_classes, height, width)): Array after applying
            softmax to logits
        image (float Tensor (num_channels, height, width)): If `None`, color-dependent
            potentials will not be added
    """
    try:
        from pydensecrf import densecrf as dcrf  # type: ignore
    except ImportError:
        raise ImportError(
            "Package pydensecrf not found. Please check installation"
            f" on https://github.com/lucasb-eyer/pydensecrf.git"
        ) from None

    num_classes, H, W = prob.shape
    dense_crf = dcrf.DenseCRF2D(W, H, num_classes)
    unary = unary_from_softmax(prob)
    dense_crf.setUnaryEnergy(unary)

    dense_crf.addPairwiseGaussian(sxy=3, compat=3)
    if image is not None:
        crf_image = (
            TF.to_dtype(image, torch.uint8, scale=True)
            .permute(1, 2, 0)
            .contiguous()
            .numpy(force=True)
        )
        dense_crf.addPairwiseBilateral(sxy=80, srgb=13, rgbim=crf_image, compat=10)

    inferenced = dense_crf.inference(iter)
    refined_prob = np.array(inferenced).reshape(num_classes, H, W)
    return refined_prob


def blur_output(output: np.ndarray, std: float = 1, **kwargs) -> np.ndarray:
    """Apply Gaussian blur on each spatial dimension separately

    Args:
        output (Tensor (num_classes, height, width)): Technically support logit and softmax
            probability
        kwargs: See `scipy.ndimage.gaussian_filter`
    """
    # https://stackoverflow.com/questions/67302611/python-gaussian-filtering-an-n-channel-image-along-only-spatial-dimensions
    sigma = (std, std, 0)
    return gaussian_filter(output, sigma, **kwargs)


def morph_pred(
    pred: np.ndarray, is_dilate: bool, skip_index: int | None = None, **kwargs
) -> dict[int, np.ndarray]:
    """Apply morphological operations on each channel separately

    Note that some pixels may have more than one prediction while some may have none

    Args:
        pred (int array (height, width)): prediction results
        is_dilate: use dilation if `True`, otherwise use erosion
        skip_index: process is skipped on that channel
        kwargs: See `scipy.ndimage.[binary_dilation,binary_erosion]`

    Returns:
        processed reults, mapping of class to binary array (height, width)
    """
    processed_pred: dict[int, np.ndarray] = {}
    classes: list[int] = np.unique(pred).tolist()  # type: ignore
    for c in classes:
        if c == skip_index:
            processed_pred[c] = pred == c
            continue

        binary = pred == c
        if is_dilate:
            processed_binary = binary_dilation(binary, **kwargs)
        else:
            processed_binary = binary_erosion(binary, **kwargs)
        processed_pred[c] = processed_binary

    return processed_pred


def threshold_prob(prob: np.ndarray, threshold=0.5) -> dict[int, np.ndarray]:
    """
    Note that some pixels may have none prediction

    Args:
        prob (float array (num_classes, height, width)): Tensor after applying
            softmax to logits
        threshold: Confidence threshold (between 0 and 1).

    Returns:
        prediction reults, mapping of class to binary array (height, width)
    """
    pred = np.argmax(prob, axis=0)
    max_prob = np.amax(prob, axis=0)
    threshold_mask = max_prob >= threshold

    thresholded_pred: dict[int, np.ndarray] = {}
    classes = np.unique(pred).tolist()
    for c in classes:
        thresholded_pred[c] = (pred == c) & threshold_mask
    return thresholded_pred


@torch.no_grad()
def inference_with_augmentations(
    model: nn.Module,
    images: Tensor,
    scales: Sequence[float] = (1,),
    fliph=False,
    flipv=False,
    rotations: Sequence[float] = (0,),
    iter_product=False,
) -> Tensor:
    """
    Args:
        images: Images after applying any preliminary augmentations
        iter_product: If `True`, all combinations of the augmentations will be tested.
            **WARNING** this will add significant time cost

    Returns:
        logits (Tensor (num_combos, batch_size, num_classes, height, width)):
            inference results of all combo
    """
    hflips = [False, True] if fliph else [False]
    vflips = [False, True] if flipv else [False]
    augment_combos = []
    if iter_product:
        augment_combos = itertools.product(scales, hflips, vflips, rotations)
    else:
        augment_combos += [(s, False, False, 0) for s in scales]
        augment_combos += [(1, h, False, 0) for h in hflips]
        augment_combos += [(1, False, v, 0) for v in vflips]
        augment_combos += [(1, False, False, r) for r in rotations]
        augment_combos = list(set(augment_combos))

    results: list[Tensor] = []
    image_size = images.shape[2:]
    for scale, hflip, vflip, rotation in augment_combos:
        # apply augmentation
        augmented_images = images.clone().detach()
        augmented_images = F.interpolate(
            augmented_images, scale_factor=scale, mode="bilinear"
        )
        if hflip:
            augmented_images = TF.horizontal_flip(augmented_images)
        if vflip:
            augmented_images = TF.vertical_flip(augmented_images)
        augmented_images = TF.rotate(augmented_images, rotation)

        # reverse augmentation except resizing back
        logits: Tensor = model(augmented_images)["out"]
        logits = TF.rotate(logits, -rotation)
        if vflip:
            logits = TF.vertical_flip(logits)
        if hflip:
            logits = TF.horizontal_flip(logits)
        logits = F.interpolate(logits, image_size, mode="bilinear")
        results.append(logits)

    return torch.stack(results)


def aggregate_outputs(
    outputs: Tensor,
    method: Literal["mode", "max", "min", "mean"],
) -> Tensor:
    """Aggregate multiple results of logits, softmax or predictions on the same inputs

    Usually, apply max/mean on logits or most on predictions

    Args:
        outputs (Tensor (num_results, batch_size, ?, height, width)): can be
            logits, softmax or predictions

    Returns:
        results (Tensor (batch_size, ?, height, width)): aggregation
    """
    match method:
        case "mode":
            return torch.mode(outputs, dim=0).values
        case "max":
            return torch.max(outputs, dim=0).values
        case "min":
            return torch.min(outputs, dim=0).values
        case "mean":
            return torch.mean(outputs, dim=0)
