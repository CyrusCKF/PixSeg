import typing
from inspect import Parameter, signature
from typing import Any, Callable, ParamSpec, TypeVar

from torch import nn

from .model_registry import SegWeights, SegWeightsEnum

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

        arg_lines = [
            "num_classes: number of output classes of the model (including the background)."
        ]

        weight_names = []
        param = sig.parameters["weights"]
        for t in typing.get_args(param.annotation):
            if isinstance(t, type) and issubclass(t, SegWeightsEnum):
                weight_names = [w.name for w in t]
        weight_line = (
            f"The pretrained weights to use. Possible values are: {weight_names}."
            if len(weight_names) > 0
            else "Not supported yet. Must be None"
        )
        arg_lines.append(f"weights: {weight_line}")

        arg_lines.append("progress: If True, display the download progress.")
        if "aux_loss" in sig.parameters:
            arg_lines.append(
                "aux_loss: If True, the model uses and returns an auxiliary loss."
            )
        if "weights_backbone" in sig.parameters:
            arg_lines.append(
                "weights_backbone: The pretrained weights for the backbone."
            )
        if (
            "kwargs" in sig.parameters
            and sig.parameters["kwargs"].kind == Parameter.VAR_KEYWORD
        ):
            arg_lines.append(
                f"**kwargs: Parameters passed to the base class {sig.return_annotation}. Please refer to the source code for more details."
            )

        doc = f"""{summary}

{sig.return_annotation.__doc__.strip()}

Args:
"""
        doc += "\n".join("    " + s for s in arg_lines)
        func.__doc__ = doc
        return func

    return wrapper
