import math
import sys
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar

import torch
from torch import Tensor, optim
from torch.optim.optimizer import ParamsT

sys.path.append(str((Path(__file__) / "..").resolve()))
from learn_api import OPTIMIZER_ZOO

T = TypeVar("T", bound=optim.Optimizer)
P = ParamSpec("P")


def register_optimizer(name: str | None = None):
    def wrapper(callable: Callable[P, T]) -> Callable[P, T]:
        key = callable.__name__ if name is None else name
        if key in OPTIMIZER_ZOO:
            raise KeyError(f"An entry is already registered under the key '{key}'.")
        OPTIMIZER_ZOO[key] = callable
        return callable

    return wrapper


register_optimizer()(optim.SGD)
register_optimizer()(optim.Adam)


@register_optimizer()
class Padam(optim.Optimizer):
    """Custom implementation of [Closing the Generalization Gap of Adaptive
    Gradient Methods in Training Deep Neural Networks](https://arxiv.org/abs/1806.06763)

    Allows partial to be negative, i.e. gradient squared in the numerator with momentum

    Note: When partial is `0.5`, this is identical to :class:`Adam/Amsgrad`. When partial is `0`,
    this is identical to :class:`SGD with momentum`.
    """

    # Reference: https://github.com/uclaml/Padam/blob/master/Padam.py

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = True,
        partial: float = 0.25,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            partial=partial,
        )
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], float] | None = None):  # type: ignore
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                p: torch.nn.Parameter
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Padam does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsgrad = group["amsgrad"]
                partial = group["partial"]

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p.data)
                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(p.data, alpha=group["weight_decay"])

                # Decay the first and second moment running average coefficient
                exp_avg: Tensor = state["exp_avg"]
                exp_avg_sq: Tensor = state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    max_exp_avg_sq: Tensor = state["max_exp_avg_sq"]
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom ** (partial * 2), value=-step_size)

        return loss
