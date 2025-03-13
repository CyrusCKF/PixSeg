"""Contain collection of strategies used to enhance inference results during test time.

See `tasks/inference.ipynb` for demo and usage
"""

import itertools
from typing import Sequence

import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF

from ..utils.transform import RandomRescale


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
        from pydensecrf.utils import unary_from_softmax
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


#####
# region Augmentations & Sliding
#####


class TestTimeAugmentations:
    def __init__(
        self,
        scales: Sequence[float] = (1,),
        hflips: Sequence[bool] = (False,),
        vflips: Sequence[bool] = (False,),
        rotations: Sequence[float] = (0,),
        iter_product=False,
    ) -> None:
        """Generate multiple transformations to enhance testtime performance

        Args:
            iter_product: If `False`, the default augmentation will be modified the values one
                at a time. If `True`, all combinations of the augmentations will be tested.
                **WARNING** this will add significant time cost.
        """
        self.augment_combos = []
        if iter_product:
            self.augment_combos = list(
                itertools.product(scales, hflips, vflips, rotations)
            )
        else:
            default_combo = (1, False, False, 0)
            mutations = [scales, hflips, vflips, rotations]
            self.augment_combos.append(default_combo)
            for i, mutation in enumerate(mutations):
                self.augment_combos += [
                    default_combo[:i] + m + default_combo[i + 1 :]
                    for m in mutation
                    if m != default_combo[i]
                ]

    def __iter__(self):
        """Iterate all combinations of augmentations and its reverse, except for resizing back"""
        for scale, hflip, vflip, rotation in self.augment_combos:
            augment = v2.Compose(
                [
                    RandomRescale((scale, scale)),
                    v2.RandomHorizontalFlip(1 if hflip else 0),
                    v2.RandomVerticalFlip(1 if vflip else 0),
                    v2.RandomRotation((rotation, rotation)),
                ]
            )
            # need to reverse the order and value
            reverse = v2.Compose(
                [
                    v2.RandomRotation((-rotation, -rotation)),
                    v2.RandomVerticalFlip(1 if vflip else 0),
                    v2.RandomHorizontalFlip(1 if hflip else 0),
                ]
            )
            yield augment, reverse


@torch.no_grad()
def inference_with_augmentations(
    model: nn.Module, images: Tensor, ttas: TestTimeAugmentations
) -> Tensor:
    """
    Args:
        images: Images after applying any preliminary augmentations

    Returns:
        logits (Tensor (num_combos, batch_size, num_classes, height, width)):
            inference results of all combo
    """
    results: list[Tensor] = []
    image_size = images.shape[2:]
    for augment, reverse in ttas:
        new_images = augment(images)
        logits: Tensor = model(new_images)["out"]
        logits = reverse(logits)
        logits = F.interpolate(logits, image_size, mode="bilinear")
        results.append(logits)

    return torch.stack(results)


def inference_with_sliding_window(
    model: nn.Module, images: Tensor, window_size: tuple[int, int]
):
    """
    Args:
        images (float Tensor (batch_size, num_channels, height, width)): Images after
            applying any preliminary augmentations

    Returns:
        logits (Tensor (num_windows, batch_size, num_classes, height, width)):
            inference results of all windows
    """
    start_indices: list[list[int]] = []  # start for each dim
    for i, window in enumerate(window_size):
        start_indices.append([])
        size = images.shape[2 + i]
        cur_idx = 0
        while cur_idx < size:
            start_indices[i].append(cur_idx)
            cur_idx += window

    results: list[Tensor] = []
    for start in itertools.product(*start_indices):
        slices = [slice(None), slice(None)]  # skip first 2 indices
        for i, (s, w) in enumerate(zip(start, window_size)):
            length = min(w, images.shape[2 + i] - s)
            slices.append(slice(s, s + length))
        cropped_image = images[slices]

        logits: Tensor = model(cropped_image)["out"]
        logits = F.interpolate(logits, cropped_image.shape[2:], mode="bilinear")
        logits_put_back = torch.zeros(logits.shape[:2] + images.shape[2:])
        logits_put_back[slices] = logits
        results.append(logits_put_back)
    return torch.stack(results)
