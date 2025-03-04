from typing import Any, TypeVar

from .model_registry import SegWeights

W = TypeVar("W", bound=SegWeights | None)


def _validate_weights_input(
    weights: W, weights_backbone: Any, num_classes: int | None
) -> tuple[W, Any, int]:
    if num_classes is None:
        num_classes = 21 if weights is None else len(weights.labels)
    if weights is not None:
        weights_backbone = None
        if num_classes != len(weights.labels):
            raise ValueError(
                f"Model weights {weights} expect number of classes"
                f"={len(weights.labels)}, but got {num_classes}"
            )
    return weights, weights_backbone, num_classes
