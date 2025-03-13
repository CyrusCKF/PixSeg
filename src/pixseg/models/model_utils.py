import typing
from enum import Enum
from inspect import signature
from typing import Any, Callable, ParamSpec, TypeVar

from torch import nn

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


T = TypeVar("T", bound=nn.Module)
P = ParamSpec("P")

_template = """{summary}

{model_desc}

Args:
    
    weights: {weights}
    progress: If True, display the download progress.
    {aux_loss}
    {backbone}
    {kwargs}
"""


def _generate_docstring(summary: str):
    """Generated doctstring can only be parsed at run time. Useful for torch.hub users"""

    def wrapper(func: Callable[P, T]) -> Callable[P, T]:
        sig = signature(func)
        assert set(sig.parameters.keys()).issuperset(
            {"num_classes", "weights", "progress"}
        )

        weight_names = []
        param = sig.parameters["weights"]
        for t in typing.get_args(param.annotation):
            if isinstance(t, type) and issubclass(t, Enum):
                weight_names = [w.name for w in t]

        backbone_weight_names = []
        if "weights_backbone" in sig.parameters:
            param = sig.parameters["weights_backbone"]
            for t in typing.get_args(param.annotation):
                if isinstance(t, type) and issubclass(t, Enum):
                    backbone_weight_names = [w.name for w in t]

        arg_desc = {
            "num_classes": "Number of output classes of the model (including the background).",
            "weights": f"The pretrained weights to use. Possible values are: {weight_names}.",
            "progress": "If True, display the download progress.",
            "aux_loss": "If True, the model uses and returns an auxiliary loss.",
            "weights_backbone": f"The pretrained weights for the backbone. Possible values are: {backbone_weight_names}.",
            "**kwargs": f"Parameters passed to the base class {sig.return_annotation}. Please refer to the source code for more details.",
        }

        if len(weight_names) == 0:
            arg_desc["weights"] = "Not supported yet. Must be None"
        if "aux_loss" not in sig.parameters:
            arg_desc.pop("aux_loss")
        if "weights_backbone" not in sig.parameters:
            arg_desc.pop("weights_backbone")
        if "kwargs" not in sig.parameters:
            arg_desc.pop("**kwargs")

        doc = f"""{summary}

{sig.return_annotation.__doc__.strip()}

Args:
"""
        doc += "\n".join(f"    {n}: {s}" for n, s in arg_desc.items())
        func.__doc__ = doc
        return func

    return wrapper
