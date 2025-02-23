from pathlib import Path
from typing import Sequence

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms import v2
from torchvision.utils import draw_segmentation_masks, make_grid


# TODO optionally not set extra_color on mask ouside the range
def draw_mask_on_image(
    image: Tensor,
    mask: Tensor,
    colors: Sequence[tuple[int, int, int]],
    extra_color=(256, 256, 256),
) -> Tensor:
    """Overlay mask on image

    Args:
        image: float Tensor of shape (3, H, W)
        mask: int Tensor of shape (H, W)
        extra_color: Used for all element in mask outside the range of :param:`colors`
    """

    num_cats = len(colors)
    mask = mask.clone().to(dtype=torch.long)
    mask[mask < 0] = num_cats
    mask[mask >= num_cats] = num_cats

    one_hot_mask = F.one_hot(mask, num_cats + 1)
    one_hot_mask = (one_hot_mask == 1).permute(2, 0, 1).contiguous()
    drawing = draw_segmentation_masks(
        image,
        one_hot_mask,
        colors=list(colors) + [extra_color],  # type: ignore
    )
    return drawing


def combine_images(images: list[Tensor], **kwargs) -> Tensor:
    """Combine images of different sizes into grid

    For acceptable kwargs, see :func:`torchvision.utils.make_grid`
    """
    largest_height = max(img.shape[-2] for img in images)
    largest_width = max(img.shape[-1] for img in images)
    padded_images = [
        v2.functional.center_crop(img, [largest_height, largest_width])
        for img in images
    ]
    grid_image = make_grid(padded_images, **kwargs)
    return grid_image


def exhibit_figure(
    figure: Figure | None = None, show: bool = False, save_to: Path | None = None
):
    if figure is None:
        figure = plt.gcf()
    if save_to is not None:
        figure.savefig(save_to)
    if show:
        plt.show()
    else:
        plt.close(figure)


def plot_confusion_matrix(
    cm: np.ndarray, labels: Sequence[str], axes: Axes | None = None
):
    if axes is None:
        axes = plt.gca()
    assert len(labels) == cm.shape[0]

    fmt = ".2g"
    if np.issubdtype(cm.dtype, np.integer):
        fmt = "d"
    elif np.issubdtype(cm.dtype, np.floating):
        fmt = ".2f"

    sns.heatmap(
        cm,
        annot=cm.shape[0] < 10,  # turn off annot when many classes
        fmt=fmt,
        cmap="YlGnBu",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes,
    )

    plt.ylabel("Ground truth")
    plt.xlabel("Prediction")
    plt.title("Confusion matrix")
    plt.tight_layout()


def plot_running_metrics(
    job_metrics: dict[str, dict[str, list[float]]], figure: Figure | None = None
):
    """Plot a 2d figure in grid to all the metrics

    Args:
        job_metrics: Mapping of job_type to sub-mapping of individual metrics to values.
            If job_type is `""`, it will not be shown in the plot title
    """
    if figure is None:
        figure = plt.gcf()

    nrows = len(job_metrics)
    ncols = max(len(v) for v in job_metrics.values())
    axes_array: np.ndarray = figure.subplots(nrows, ncols, squeeze=False)
    figure.set_size_inches(ncols * 3, nrows * 3)

    for i, job in enumerate(sorted(job_metrics.keys())):
        metrics = job_metrics[job]
        for j, m in enumerate(sorted(metrics.keys())):
            values = metrics[m]
            axes: Axes = axes_array[i, j]
            title = job + "/" + m if job != "" else m
            axes.set_title(title)
            axes.set_xlabel("Step")
            axes.set_ylabel(m)
            axes.plot(values)

        # remove excessive axes
        for j in range(len(metrics), ncols):
            figure.delaxes(axes_array[i, j])
    figure.tight_layout()


def _test():
    from sklearn.metrics import confusion_matrix

    job_metrics = {
        "train": {"loss": [1, 2, 3, 4], "acc": [0.4, 0.51, 0.63, 0.6543]},
        "val": {"acc": [0.42, 0.39, 0.53, 0.5543]},
    }
    plot_running_metrics(job_metrics)
    exhibit_figure(show=True)

    num_classes = 20
    truths = np.random.randint(0, num_classes, [10000])
    preds = np.random.randint(0, num_classes, [10000])
    matrix = confusion_matrix(truths, preds, labels=range(num_classes))
    matrix /= matrix.sum(axis=1, keepdims=True)  # normalize
    plot_confusion_matrix(matrix, [f"Group {i}" for i in range(num_classes)])
    exhibit_figure(show=True)


if __name__ == "__main__":
    _test()
