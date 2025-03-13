from collections import defaultdict
from typing import cast

import numpy as np
import torch
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
        """Values outside the range of `[0, num_classes)` will be ignored"""
        results_cm = fast_confusion_matrix(truths, preds, self.num_classes)
        self.confusion_matrix += results_cm.numpy()

    def store_measures(self, num_data: int, measures: dict[str, float]):
        """Expect the measures in "sum" of all data"""
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

    class_accs: np.ndarray = TP / (TP + FN + epsilon)
    mean_acc: np.ndarray = class_accs.mean()
    metrics["macc"] = cast(float, mean_acc.item())

    class_ious = TP / (TP + FP + FN + epsilon)
    mean_iou: np.ndarray = class_ious.mean()
    metrics["miou"] = cast(float, mean_iou.item())

    frequency = (TP + FN) / cm.sum()
    fwiou: np.ndarray = (class_ious * frequency).sum()
    metrics["fwiou"] = cast(float, fwiou.item())

    class_dices = 2 * TP / (2 * TP + FP + FN + epsilon)
    mean_dice: np.ndarray = class_dices.mean()
    metrics["dice"] = cast(float, mean_dice.item())

    return metrics


def fast_confusion_matrix(truths: Tensor, preds: Tensor, num_classes: int) -> Tensor:
    """Calculate the confusion matrix in Tensor

    This function is faster than :module:`sklearn.metrics` since it doesn't convert
    to numpy array every single time. All indices outside the range of `[0, num_classes)`
    are ignored.

    Returns:
        A confusion matrix *(int Tensor (N, N))* where element at `[i, j]` is equal to
            the number of ground truths in class `i` and predicted to be class `j`
    """
    truths = truths.detach().cpu().flatten()
    preds = preds.detach().cpu().flatten()
    in_range = (
        (truths >= 0) & (truths < num_classes) & (preds >= 0) & (preds < num_classes)
    )
    truths = truths[in_range]
    preds = preds[in_range]
    # combine to single tensor first to get frequency
    indices = truths * num_classes + preds
    matrix = torch.bincount(indices, minlength=num_classes * num_classes)
    matrix = matrix.reshape(num_classes, num_classes)
    return matrix


def _test():
    num_classes = 10
    truths = torch.randint(0, num_classes, [160, 90]).flatten()
    preds = torch.randint(0, num_classes, [160, 90]).flatten()
    matrix = fast_confusion_matrix(truths, preds, num_classes).numpy()
    print(matrix)
    print(metrics_from_confusion(matrix))

    print("\nMetricStore -----")
    ms = MetricStore(5)
    truths = torch.randint(0, 4, [100, 50])
    preds = torch.randint(0, 4, [100, 50])
    ms.store_results(truths, preds)
    print(ms.summarize())

    from timeit import default_timer

    from sklearn.metrics import confusion_matrix

    truths = torch.randint(0, num_classes, [1600, 900]).to("cuda").flatten()
    preds = torch.randint(0, num_classes, [1600, 900]).to("cuda").flatten()

    start_time = default_timer()
    for i in range(100):
        fast_confusion_matrix(truths, preds, num_classes).numpy(force=True)
    end_time = default_timer()
    print("fast_confusion_matrix", end_time - start_time)

    start_time = default_timer()
    for i in range(100):
        confusion_matrix(
            truths.numpy(force=True), preds.numpy(force=True), labels=range(num_classes)
        )
    end_time = default_timer()
    print("sklearn.metrics", end_time - start_time)


if __name__ == "__main__":
    _test()
