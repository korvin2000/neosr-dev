import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from neosr.config import ConfigError
from neosr.optimizers.adamw_sf import swap_sf_weights
from neosr.optimizers.adan import adan_group_step, clip_grad_norm_

_SF_STATE = ("z", "exp_avg_sq", "exp_avg_diff", "neg_pre_grad")


def _multi_tensor_sf_adan(
    params: list[Tensor],
    grads: list[Tensor],
    z: list[Tensor],
    exp_avg_sqs: list[Tensor],
    exp_avg_diffs: list[Tensor] = (),  # type: ignore[assignment]
    neg_pre_grads: list[Tensor] = (),  # type: ignore[assignment]
    *,
    beta1: float,
    beta2: float,
    beta3: float,
    bias_correction2: float,
    lr: float,
    ckp1: float,
    weight_decay: float,
    eps: float,
) -> None:
    """Schedule-free Adan step; grads are overwritten with the direction U (as adamw_sf does).
    With beta2 = 0 these are exactly adamw_sf's ops with betas (beta1, beta3)."""
    u = grads
    if beta2:
        torch._foreach_add_(neg_pre_grads, grads)  # d_t = g_t - g_{t-1}
        torch._foreach_lerp_(exp_avg_diffs, neg_pre_grads, 1 - beta2)  # v_t
        torch._foreach_mul_(neg_pre_grads, beta2)
        torch._foreach_add_(neg_pre_grads, grads)  # u_t = g_t + b2 d_t
        u = neg_pre_grads
    torch._foreach_mul_(exp_avg_sqs, beta3)
    torch._foreach_addcmul_(exp_avg_sqs, u, u, value=1 - beta3)  # n_t
    denom = torch._foreach_sqrt(exp_avg_sqs)
    torch._foreach_add_(denom, eps)
    if beta2:
        torch._foreach_copy_(neg_pre_grads, grads)
        torch._foreach_neg_(neg_pre_grads)  # -g_t for the next step
        torch._foreach_add_(grads, exp_avg_diffs, alpha=beta2 / bias_correction2)
    torch._foreach_div_(grads, denom)
    if weight_decay != 0:
        torch._foreach_add_(grads, params, alpha=weight_decay)  # decay at the old y
    # y <- (1-b1) z + b1 x in place, without computing x
    torch._foreach_lerp_(params, z, weight=ckp1)
    torch._foreach_add_(params, grads, alpha=lr * (beta1 * (1 - ckp1) - 1))
    torch._foreach_sub_(z, grads, alpha=lr)


class adan_sf(Optimizer):
    """Schedule-Free (https://arxiv.org/abs/2405.15682) Adan (https://arxiv.org/abs/2208.06677).

    Schedule-free mode, per step with g_t the (clipped) gradient at the train weights y_t:
        d_t = g_t - g_{t-1} (d_1 = 0),  v_t = b2 v + (1-b2) d_t,  u_t = g_t + b2 d_t,
        n_t = b3 n + (1-b3) u_t^2,  gamma_t = lr * min(t / warmup_steps, 1) * sqrt(1 - b3^t),
        U_t = (g_t + b2 / (1 - b2^t) v_t) / (sqrt(n_t) + eps) + weight_decay * y_t,
        z <- z - gamma_t U_t,  x <- (1 - c_t) x + c_t z,  y = (1 - b1) z + b1 x,
    with c_t the weight_lr_power/r averaging weight of adamw_sf. Adan's first moment is replaced
    by the schedule-free interpolation (b1). betas[1] = 0 drops the difference term and is
    exactly adamw_sf with betas (b1, b3). schedule_free=False is plain Adan (no_prox) with warmup.

    State per parameter: z, exp_avg_sq, plus exp_avg_diff and neg_pre_grad when betas[1] > 0.
    Schedule-free groups carry sf_version; states of the old recurrence (no sf_version) are refused.

    This optimizer requires that .train() and .eval() be called before the
    beginning of training and evaluation respectively. The optimizer should
    also be placed in eval mode when saving checkpoints.

    Arguments:
    ---------
        params (iterable): iterable of parameters to optimize or
            dicts defining parameter groups.
        lr (float, optional): learning rate. (default: 1.6e-3)
        betas (Tuple[float, float, float], optional): interpolation (b1),
            difference EMA (b2) and second moment (b3). (default: (0.98, 0.92, 0.99))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): decoupled weight decay
            (L2 penalty) (default: 0.02)
        max_grad_norm (float, optional): value used to clip
            global grad norm (default: 0.0 no clip; the trainer
            already clips with train.grad_clip)
        warmup_steps (int): Enables a linear learning rate warmup (default: 0).
        r (float): Use polynomial weighting in the average
            with power r (default: 0).
        weight_lr_power (float): During warmup, the weights in the average will
            be equal to lr raised to this power. Set to 0 for no weighting
            (default: 2.0).
        schedule_free (bool): Whether to enable Schedule-Free (default: True)

    """

    # 2: z takes the preconditioned step (1, unversioned: raw-gradient z, Adan first moment)
    STATE_VERSION = 2

    def __init__(  # type: ignore[no-untyped-def]
        self,
        params: Iterable[Tensor],
        lr: float = 1.6e-3,
        betas: tuple[float, float, float] = (0.98, 0.92, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.02,
        max_grad_norm: float = 0.0,
        warmup_steps: int = 0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        schedule_free: bool = True,
    ) -> None:
        if not max_grad_norm >= 0.0:
            msg = f"Invalid Max grad norm: {max_grad_norm}"
            raise ValueError(msg)
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "r": r,
            "weight_decay": weight_decay,
            "max_grad_norm": max_grad_norm,
            "warmup_steps": warmup_steps,
            "train_mode": True,
            "weight_sum": 0.0,
            "lr_max": -1.0,
            "weight_lr_power": weight_lr_power,
            "schedule_free": schedule_free,
        }
        super().__init__(params, defaults)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        # validate the effective options: groups may override the constructor defaults
        group = {**self.defaults, **param_group}
        if not group["lr"] >= 0.0:
            msg = f"Invalid learning rate: {group['lr']}"
            raise ValueError(msg)
        if not group["eps"] >= 0.0:
            msg = f"Invalid epsilon value: {group['eps']}"
            raise ValueError(msg)
        beta1, beta2, beta3 = group["betas"]
        # eval() divides by beta1
        if not 0.0 < beta1 < 1.0:
            msg = f"Invalid beta parameter at index 0: {beta1}"
            raise ValueError(msg)
        if not 0.0 <= beta2 < 1.0:
            msg = f"Invalid beta parameter at index 1: {beta2}"
            raise ValueError(msg)
        if not 0.0 <= beta3 < 1.0:
            msg = f"Invalid beta parameter at index 2: {beta3}"
            raise ValueError(msg)
        if group["schedule_free"]:
            param_group["sf_version"] = self.STATE_VERSION
        super().add_param_group(param_group)

    def eval(self) -> None:
        swap_sf_weights(self, train=False)

    def train(self) -> None:
        swap_sf_weights(self, train=True)

    @torch.no_grad()
    def step(self, closure: Callable[..., Any] | None = None) -> float:  # type: ignore[reportIncompatibleMethodOverride,override]
        """Performs a single optimization step."""
        if any(g["schedule_free"] and not g["train_mode"] for g in self.param_groups):
            msg = "Not in train mode!"
            raise ValueError(msg)

        loss = 0.0
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        clip_grad_norm_(self.param_groups, self.defaults["max_grad_norm"])
        for group in self.param_groups:
            step = group["step"] = group.get("step", 0) + 1
            warmup_steps = group["warmup_steps"]
            sched = step / warmup_steps if step < warmup_steps else 1.0
            if not group["schedule_free"]:
                adan_group_step(
                    self.state, group, group["lr"] * sched, no_prox=True, foreach=True
                )
                continue

            beta1, beta2, beta3 = group["betas"]
            # gamma_t: the bias correction of n is folded into the step, as in adamw_sf
            lr = group["lr"] * sched * math.sqrt(1.0 - beta3**step)
            lr_max = group["lr_max"] = max(lr, group["lr_max"])
            weight = (step ** group["r"]) * (lr_max ** group["weight_lr_power"])
            weight_sum = group["weight_sum"] = group["weight_sum"] + weight
            ckp1 = weight / weight_sum if weight_sum else 0.0

            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            for p in params:
                state = self.state[p]
                if "z" not in state:
                    state["z"] = torch.clone(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                if beta2 and "neg_pre_grad" not in state:
                    state["exp_avg_diff"] = torch.zeros_like(p)
                    state["neg_pre_grad"] = -p.grad  # d_1 = 0
            states = [self.state[p] for p in params]
            keys = _SF_STATE if beta2 else _SF_STATE[:2]

            _multi_tensor_sf_adan(
                params,
                [p.grad for p in params],
                *([s[k] for s in states] for k in keys),
                beta1=beta1,
                beta2=beta2,
                beta3=beta3,
                bias_correction2=1.0 - beta2**step,
                lr=lr,
                ckp1=ckp1,
                weight_decay=group["weight_decay"],
                eps=group["eps"],
            )

        return loss

    def __setstate__(self, state: dict[str, Any]) -> None:
        # runs for load_state_dict and unpickling, before the optimizer is touched
        for group in state["param_groups"]:
            if group.setdefault("schedule_free", True):
                self._check_sf_group(group, state["state"])
        super().__setstate__(state)

    @classmethod
    def _check_sf_group(cls, group: dict[str, Any], state: dict) -> None:
        hint = "; start a new run with pretrain_network_* instead of resuming"
        if (version := group.get("sf_version", 1)) != cls.STATE_VERSION:
            msg = (
                f"adan_sf: schedule-free state version {version!r}, this adan_sf needs "
                f"{cls.STATE_VERSION} (the recurrence changed){hint}"
            )
            raise ConfigError(msg)
        need = _SF_STATE if group["betas"][1] else _SF_STATE[:2]
        for i, p in enumerate(group["params"]):
            # an empty state is a parameter that has not been stepped yet
            if (s := state.get(p)) and (missing := [k for k in need if k not in s]):
                msg = f"adan_sf: schedule-free state of parameter #{i} lacks {missing}{hint}"
                raise ConfigError(msg)
