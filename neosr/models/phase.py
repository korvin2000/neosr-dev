"""Training-step building blocks: one Phase per optimizer, the generator EMA and the log meter."""

from __future__ import annotations

import copy
import math
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
from torch import distributed as dist
from torch.nn.utils import clip_grads_with_norm_, get_total_norm

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from torch import Tensor, nn
    from torch.optim import Optimizer

    from neosr.losses.base import Context, Loss


class Phase:
    """One network's objective: terms, optimizer, fp16 scaler, clipping, schedulers, skip accounting."""

    def __init__(
        self,
        name: str,
        params: list[nn.Parameter],
        opt: Optimizer,
        terms: list[Loss],
        *,
        amp_dtype: torch.dtype,
        use_amp: bool,
        accumulate: int,
        grad_clip: float,
        max_skip_streak: int = 10,
        schedulers: Iterable = (),
    ) -> None:
        self.name, self.params, self.opt, self.terms = name, params, opt, terms
        self.autocast = {"device_type": "cuda", "dtype": amp_dtype, "enabled": use_amp}
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=use_amp and amp_dtype is torch.float16, init_scale=2.0**5
        )
        self.inv_accum, self.grad_clip = 1.0 / accumulate, grad_clip
        self.max_skip_streak, self.schedulers = max_skip_streak, list(schedulers)
        # host ints: skipped in the log window, consecutive skips
        self.skipped = self.streak = 0
        # device scalar of the last successful update
        self.grad_norm: Tensor | None = None

    def backward(self, ctx: Context) -> None:
        with torch.autocast(**self.autocast):
            total = sum(t.compute(ctx) for t in self.terms)
        ctx.log(f"l_{self.name}_total", total)  # undivided micro-batch objective
        self.scaler.scale(total * self.inv_accum).backward(inputs=self.params)

    def step(self) -> bool:
        self.scaler.unscale_(self.opt)  # no-op unless fp16
        norm = get_total_norm(
            [p.grad for p in self.params if p.grad is not None], foreach=True
        )
        if self.grad_clip:
            clip_grads_with_norm_(self.params, self.grad_clip, norm, foreach=True)
        # the one host sync per update; rank-identical under DDP
        ok = bool(norm.isfinite())
        if ok:
            # fp16: re-checks found_inf on the drained stream
            self.scaler.step(self.opt)
            self.grad_norm = norm
        self.scaler.update()  # fp16: halves the scale after an inf/NaN window
        self.opt.zero_grad(set_to_none=True)
        for s in self.schedulers:  # clock = attempted updates
            s.step()
        self.skipped += not ok
        self.streak = 0 if ok else self.streak + 1
        if self.streak >= self.max_skip_streak:
            raise FloatingPointError(
                f"net_{self.name}: {self.streak} consecutive non-finite updates; "
                "restart from the last checkpoint with a lower learning rate"
            )
        return ok


def no_sync(net: nn.Module, boundary: bool):
    """DDP: reduce only on the window's last micro-batch."""
    return net.no_sync() if not boundary and hasattr(net, "no_sync") else nullcontext()


class Ema:
    """EMA of the bare eager G. First successful update copies (AveragedModel semantics); later ones lerp."""

    def __init__(self, net: nn.Module, decay: float) -> None:
        self.net, self.w, self.lists = net, 1.0 - decay, None
        self.module = copy.deepcopy(net).eval().requires_grad_(False)

    def _bind(self) -> None:
        self.lists = (
            [*self.module.parameters()],
            [*self.net.parameters()],
            [*self.module.buffers()],
            [*self.net.buffers()],
        )

    @torch.no_grad()
    def update(self) -> None:  # called only after a successful G update
        if self.lists is None:
            # load_state_dict also resizes per-channel QAT observers
            self.module.load_state_dict(self.net.state_dict())
            self._bind()
            return
        dp, sp, db, sb = self.lists
        torch._foreach_lerp_(dp, sp, self.w)
        if db:
            torch._foreach_copy_(db, sb)


class LossMeter:
    """Detached device sums with host counts per key; one stack + all_reduce + tolist per read."""

    def __init__(self) -> None:
        self.sum: dict[str, Tensor] = {}
        self.n: dict[str, int] = {}

    def add(self, logs: Mapping[str, Tensor]) -> None:
        for k, v in logs.items():
            s = self.sum.get(k)
            self.sum[k] = v.float().clone() if s is None else s.add_(v)
            self.n[k] = self.n.get(k, 0) + 1

    def read(self, reset: bool = True) -> dict[str, float]:
        keys = list(self.sum)
        if not keys:
            return {}
        vec = torch.stack([self.sum[k] / self.n[k] for k in keys])
        if dist.is_initialized():
            dist.all_reduce(vec, op=dist.ReduceOp.AVG)  # collective: call on all ranks
        out = dict(zip(keys, vec.tolist(), strict=True))
        if reset:
            self.sum.clear()
            self.n.clear()
        if bad := [k for k, v in out.items() if not math.isfinite(v)]:
            raise FloatingPointError(
                f"non-finite training values {bad}; nothing was saved from this window"
            )
        return out
