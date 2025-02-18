from collections import defaultdict
from typing import cast

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import Tensor


class MetricStore:
    """Accumulate batch prediction results and compute metrics efficiently

    Example usage:
    ```
        ms = MetricStore(10)
        foreach iter:
            ms.store_results(predictions, ground_truths)
            ms.store_measures(batch_size, { "loss": loss })
        metrics = ms.summarize()
    ```
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.confusion_matrix: np.ndarray = np.zeros(
            [num_classes, num_classes], dtype=np.int_
        )

        # store other useful info
        self.count_data: int = 0
        self.measures: dict[str, float] = defaultdict(lambda: 0)

    def store_results(self, truths: Tensor, preds: Tensor):
        """Values outside the range of `(0, num_classes)` will be ignored"""
        true_arr = truths.numpy(force=True).flatten()
        pred_arr = preds.numpy(force=True).flatten()
        # confusion_matrix will ignore values outside the range
        results_cm = confusion_matrix(
            true_arr, pred_arr, labels=range(self.num_classes)
        )
        self.confusion_matrix += results_cm

    def store_measures(self, num_data: int, measures: dict[str, float]):
        self.count_data += num_data
        for k, v in measures.items():
            self.measures[k] += v

    def summarize(self) -> dict[str, float]:
        """Return the average metrics and measures

        See :func:`metrics_from_confusion` for all the computed metrics
        """
        if len(self.measures) > 0 and self.count_data == 0:
            raise ValueError("Number of data stored is 0")
        avg_measures = {k: v / self.count_data for k, v in self.measures.items()}
        return metrics_from_confusion(self.confusion_matrix) | avg_measures


def metrics_from_confusion(cm: np.ndarray) -> dict[str, float]:
    """Calculate metrics from confusion matrix

    Confusion matrix should not be normalized and is an int array of
    shape (num_classes, num_classes)

    Returns:
        A dictionary of scores
        - "acc": pixel accuracy
        - "macc": mean pixel accuracies, aka mean recalls
        - "miou": mean intersection over union, aka Jaccard
        - "fwiou": frequency weighted iou
        - "dice": (hard) Dice score, aka macro average of F1
    """
    metrics: dict[str, float] = {}
    TP: np.ndarray = np.diag(cm)
    FP: np.ndarray = cm.sum(axis=0) - TP
    FN: np.ndarray = cm.sum(axis=1) - TP
    epsilon = 1e-6  # prevent division by zero

    acc: np.ndarray = TP.sum() / cm.sum()
    metrics["acc"] = cast(float, acc.item())

    cat_accs: np.ndarray = TP / (TP + FN + epsilon)
    mean_acc: np.ndarray = cat_accs.mean()
    metrics["macc"] = cast(float, mean_acc.item())

    cat_ious = TP / (TP + FP + FN + epsilon)
    mean_iou: np.ndarray = cat_ious.mean()
    metrics["miou"] = cast(float, mean_iou.item())

    frequency = (TP + FN) / cm.sum()
    fwiou: np.ndarray = (cat_ious * frequency).sum()
    metrics["fwiou"] = cast(float, fwiou.item())

    cat_dices = 2 * TP / (2 * TP + FP + FN + epsilon)
    mean_dice: np.ndarray = cat_dices.mean()
    metrics["dice"] = cast(float, mean_dice.item())

    return metrics


def _test():
    num_cats = 10
    truths = np.random.randint(0, num_cats, [160, 90]).flatten()
    preds = np.random.randint(0, num_cats, [160, 90]).flatten()
    matrix = confusion_matrix(truths, preds)
    print(matrix)
    print(metrics_from_confusion(matrix))

    print("\nMetricStore -----")
    ms = MetricStore(5)
    truths = torch.randint(0, 4, [100, 50])
    preds = torch.randint(0, 4, [100, 50])
    ms.store_results(truths, preds)
    print(ms.summarize())


if __name__ == "__main__":
    _test()
