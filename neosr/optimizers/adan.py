import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


def _single_tensor_adan(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    exp_avg_diffs: list[Tensor],
    neg_pre_grads: list[Tensor],
    *,
    beta1: float,
    beta2: float,
    beta3: float,
    bias_correction1: float,
    bias_correction2: float,
    bias_correction3_sqrt: float,
    lr: float,
    weight_decay: float,
    eps: float,
    no_prox: bool,
) -> None:
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        exp_avg_diff = exp_avg_diffs[i]
        neg_grad_or_diff = neg_pre_grads[i]

        # for memory saving, we use `neg_grad_or_diff`
        # to get some temp variable in a inplace way
        neg_grad_or_diff.add_(grad)

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # m_t
        exp_avg_diff.mul_(beta2).add_(neg_grad_or_diff, alpha=1 - beta2)  # diff_t

        neg_grad_or_diff.mul_(beta2).add_(grad)
        exp_avg_sq.mul_(beta3).addcmul_(
            neg_grad_or_diff, neg_grad_or_diff, value=1 - beta3
        )  # n_t

        denom = ((exp_avg_sq).sqrt() / bias_correction3_sqrt).add_(eps)
        step_size_diff = lr * beta2 / bias_correction2
        step_size = lr / bias_correction1

        if no_prox:
            param.mul_(1 - lr * weight_decay)
            param.addcdiv_(exp_avg, denom, value=-step_size)
            param.addcdiv_(exp_avg_diff, denom, value=-step_size_diff)
        else:
            param.addcdiv_(exp_avg, denom, value=-step_size)
            param.addcdiv_(exp_avg_diff, denom, value=-step_size_diff)
            param.div_(1 + lr * weight_decay)

        neg_grad_or_diff.zero_().add_(grad, alpha=-1.0)


def _multi_tensor_adan(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    exp_avg_diffs: list[Tensor],
    neg_pre_grads: list[Tensor],
    *,
    beta1: float,
    beta2: float,
    beta3: float,
    bias_correction1: float,
    bias_correction2: float,
    bias_correction3_sqrt: float,
    lr: float,
    weight_decay: float,
    eps: float,
    no_prox: bool,
) -> None:
    if len(params) == 0:
        return

    # for memory saving, we use `neg_pre_grads`
    # to get some temp variable in a inplace way
    torch._foreach_add_(neg_pre_grads, grads)

    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)  # m_t

    torch._foreach_mul_(exp_avg_diffs, beta2)
    torch._foreach_add_(exp_avg_diffs, neg_pre_grads, alpha=1 - beta2)  # diff_t

    torch._foreach_mul_(neg_pre_grads, beta2)
    torch._foreach_add_(neg_pre_grads, grads)
    torch._foreach_mul_(exp_avg_sqs, beta3)
    torch._foreach_addcmul_(
        exp_avg_sqs, neg_pre_grads, neg_pre_grads, value=1 - beta3
    )  # n_t

    denom = torch._foreach_sqrt(exp_avg_sqs)
    torch._foreach_div_(denom, bias_correction3_sqrt)
    torch._foreach_add_(denom, eps)

    step_size_diff = lr * beta2 / bias_correction2
    step_size = lr / bias_correction1

    if no_prox:
        torch._foreach_mul_(params, 1 - lr * weight_decay)
        torch._foreach_addcdiv_(params, exp_avgs, denom, value=-step_size)
        torch._foreach_addcdiv_(params, exp_avg_diffs, denom, value=-step_size_diff)
    else:
        torch._foreach_addcdiv_(params, exp_avgs, denom, value=-step_size)
        torch._foreach_addcdiv_(params, exp_avg_diffs, denom, value=-step_size_diff)
        torch._foreach_div_(params, 1 + lr * weight_decay)
    torch._foreach_zero_(neg_pre_grads)
    torch._foreach_add_(neg_pre_grads, grads, alpha=-1.0)


_ADAN_STATE = ("exp_avg", "exp_avg_sq", "exp_avg_diff", "neg_pre_grad")


def clip_grad_norm_(groups: list[dict[str, Any]]) -> None:
    """Optional optimizer-level clip by each group's max_grad_norm (0 = off; the neosr trainer already
    clips with train.grad_clip) against the norm over all groups: one _foreach_norm and a 0-dim
    coefficient per group; no .item(), so no host sync. Per group, so a resumed group value applies."""
    if all(g["max_grad_norm"] <= 0 for g in groups):
        return
    grads = [[p.grad for p in g["params"] if p.grad is not None] for g in groups]
    if not any(grads):
        return
    norm = torch.linalg.vector_norm(
        torch.stack(torch._foreach_norm([t for gs in grads for t in gs]))
    )
    for g, gs in zip(groups, grads, strict=True):
        if gs and (max_norm := g["max_grad_norm"]) > 0:
            torch._foreach_mul_(gs, torch.clamp(max_norm / (norm + g["eps"]), max=1.0))


def adan_group_step(
    state: dict[Tensor, dict[str, Tensor]],
    group: dict[str, Any],
    lr: float,
    no_prox: bool,
    foreach: bool,
) -> None:
    """One plain Adan step of a group whose "step" is already advanced; shared with adan_sf(schedule_free=False)."""
    params = [p for p in group["params"] if p.grad is not None]
    if not params:
        return
    for p in params:
        s = state[p]
        if len(s) == 0:
            s["exp_avg"] = torch.zeros_like(p)
            s["exp_avg_sq"] = torch.zeros_like(p)
            s["exp_avg_diff"] = torch.zeros_like(p)
        if "neg_pre_grad" not in s or group["step"] == 1:
            s["neg_pre_grad"] = -p.grad
    states = [state[p] for p in params]
    beta1, beta2, beta3 = group["betas"]
    (_multi_tensor_adan if foreach else _single_tensor_adan)(
        params,
        [p.grad for p in params],
        *([s[k] for s in states] for k in _ADAN_STATE),
        beta1=beta1,
        beta2=beta2,
        beta3=beta3,
        bias_correction1=1.0 - beta1 ** group["step"],
        bias_correction2=1.0 - beta2 ** group["step"],
        bias_correction3_sqrt=math.sqrt(1.0 - beta3 ** group["step"]),
        lr=lr,
        weight_decay=group["weight_decay"],
        eps=group["eps"],
        no_prox=no_prox,
    )


class adan(Optimizer):
    """'Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models':
        https://arxiv.org/abs/2208.06677.

    Arguments:
    ---------
        params (iterable): iterable of parameters to optimize or
            dicts defining parameter groups.
        lr (float, optional): learning rate. (default: 5e-4)
        betas (Tuple[float, float, float], optional): coefficients used for
            first- and second-order moments. (default: (0.98, 0.92, 0.99))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): decoupled weight decay
            (L2 penalty) (default: 0.02)
        max_grad_norm (float, optional): value used to clip
            global grad norm (default: 0.0 no clip; the trainer
            already clips with train.grad_clip)
        no_prox (bool): how to perform the decoupled weight decay
            (default: True)
        foreach (bool): if True would use torch._foreach implementation.
            It's faster but uses slightly more memory. (default: True)

    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 5e-4,
        betas: tuple[float, float, float] = (0.98, 0.92, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.02,
        max_grad_norm: float = 0.0,
        no_prox: bool = True,
        foreach: bool = True,
    ) -> None:
        if not max_grad_norm >= 0.0:
            msg = f"Invalid Max grad norm: {max_grad_norm}"
            raise ValueError(msg)
        if not lr >= 0.0:
            msg = f"Invalid learning rate: {lr}"
            raise ValueError(msg)
        if not eps >= 0.0:
            msg = f"Invalid epsilon value: {eps}"
            raise ValueError(msg)
        if not 0.0 <= betas[0] < 1.0:
            msg = f"Invalid beta parameter at index 0: {betas[0]}"
            raise ValueError(msg)
        if not 0.0 <= betas[1] < 1.0:
            msg = f"Invalid beta parameter at index 1: {betas[1]}"
            raise ValueError(msg)
        if not 0.0 <= betas[2] < 1.0:
            msg = f"Invalid beta parameter at index 2: {betas[2]}"
            raise ValueError(msg)

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "max_grad_norm": max_grad_norm,
            "no_prox": no_prox,
            "foreach": foreach,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[..., Any] | None = None) -> float:  # type: ignore[reportIncompatibleMethodOverride,override]
        """Performs a single optimization step."""
        loss = 0.0
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        clip_grad_norm_(self.param_groups)
        for group in self.param_groups:
            # assume same step across group now to simplify things
            # per parameter step can be easily support
            # by making it tensor, or pass list into kernel
            group["step"] = group.get("step", 0) + 1
            adan_group_step(
                self.state, group, group["lr"], group["no_prox"], group["foreach"]
            )

        return loss

    def __setstate__(self, state: dict[str, bool]) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("no_prox", False)
