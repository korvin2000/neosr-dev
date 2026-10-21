"""Training-step building blocks: one Phase per optimizer, the generator EMA and the log meter."""

from __future__ import annotations

import copy
import math
from contextlib import nullcontext
from itertools import zip_longest
from typing import TYPE_CHECKING

import torch
from torch import distributed as dist
from torch.nn.utils import clip_grads_with_norm_, get_total_norm

from neosr.config import ConfigError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from logging import Logger

    from torch import Tensor, nn
    from torch.optim import Optimizer

    from neosr.losses.base import Context, Loss


def trainable(net: nn.Module) -> tuple[list[str], list[nn.Parameter]]:
    """Ordered names and parameters that require grad: what the optimizer owns."""
    pairs = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
    return [n for n, _ in pairs], [p for _, p in pairs]


def is_schedule_free(opt: Optimizer) -> bool:
    """adamw_sf is always SF; adan_sf honours its flag; plain optimizers have no eval()."""
    return callable(getattr(opt, "eval", None)) and bool(
        opt.defaults.get("schedule_free", True)
    )


def _first_diff(a: Sequence, b: Sequence) -> tuple[int, object, object]:
    return next((i, x, y) for i, (x, y) in enumerate(zip_longest(a, b)) if x != y)


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
        names: Sequence[str] | None = None,
        opt_type: str | None = None,
    ) -> None:
        self.name, self.params, self.opt, self.terms = name, params, opt, terms
        # resume identity: config optimizer type, SF mode, ordered trainable names;
        # no positional fallback: same-count swaps would silently remap optimizer moments
        self.opt_type, self.sf = opt_type or type(opt).__name__, is_schedule_free(opt)
        self.names = None if names is None else list(names)
        if self.names is not None and len(self.names) != len(params):
            msg = f"net_{name}: {len(self.names)} names for {len(params)} parameters"
            raise ValueError(msg)
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

    def _need_names(self) -> list[str]:
        if self.names is None:
            msg = f"net_{self.name}: state needs the trainable names (pass names=)"
            raise RuntimeError(msg)
        return self.names

    def state_dict(self) -> dict:
        return {
            "optimizer": {
                "type": self.opt_type,
                "sf": self.sf,
                "names": self._need_names(),
                "state": self.opt.state_dict(),
            },
            "scaler": self.scaler.state_dict(),  # {} when fp16 is off
            "schedulers": [
                {"type": type(s).__name__, "state": s.state_dict()}
                for s in self.schedulers
            ],
        }

    def load_state_dict(self, sd: Mapping, log: Logger) -> None:
        """Strict resume: a changed optimizer, SF mode, trainable set or scheduler list raises."""
        o, hint = sd["optimizer"], "; start a new run with pretrain_network_*"
        if (o["type"], o["sf"]) != (self.opt_type, self.sf):
            msg = (
                f"net_{self.name}: optimizer {o['type']}(sf={o['sf']}) -> "
                f"{self.opt_type}(sf={self.sf}){hint}"
            )
            raise ConfigError(msg)
        if o["names"] != (names := self._need_names()):
            i, a, b = _first_diff(o["names"], names)
            msg = f"net_{self.name}: trainable parameter #{i} changed: {a} -> {b}{hint}"
            raise ConfigError(msg)
        saved = [s["type"] for s in sd["schedulers"]]
        if saved != (now := [type(s).__name__ for s in self.schedulers]):
            i, a, b = _first_diff(saved, now)
            msg = f"net_{self.name}: scheduler #{i} changed: {a} -> {b}{hint}"
            raise ConfigError(msg)
        self.opt.load_state_dict(o["state"])  # restores SF train_mode group flags
        for s, ss in zip(self.schedulers, sd["schedulers"], strict=True):
            s.load_state_dict(ss["state"])
        if self.scaler.is_enabled():
            if sd["scaler"]:
                self.scaler.load_state_dict(sd["scaler"])
            else:
                log.warning(f"net_{self.name}: fp16 GradScaler starts fresh")
        elif sd["scaler"]:
            log.warning(f"net_{self.name}: saved GradScaler state ignored (fp16 off)")


def no_sync(net: nn.Module, boundary: bool):
    """DDP: reduce only on the window's last micro-batch."""
    return net.no_sync() if not boundary and hasattr(net, "no_sync") else nullcontext()


class Ema:
    """EMA of the bare eager G. First successful update copies (AveragedModel semantics); later ones lerp.
    With `power` (ema_pytorch warmup) the k-th lerp after the copy uses min(1 - (1 + k)**-power, decay);
    the lerp count is then saved so that resume stays exact. A state without it counts as warmed up."""

    KEY = "@ema_lerps"  # not a module path: never collides with a parameter or buffer name

    def __init__(
        self, net: nn.Module, decay: float, power: float | None = None
    ) -> None:
        self.net, self.decay, self.power, self.lists = net, decay, power, None
        self.lerps: int | None = 0  # None: warmed up (state saved without the count)
        self.module = copy.deepcopy(net).eval().requires_grad_(False)

    def _bind(self) -> None:
        self.lists = (
            [*self.module.parameters()],
            [*self.net.parameters()],
            [*self.module.buffers()],
            [*self.net.buffers()],
        )

    def current_decay(self) -> float:
        if self.power is None or self.lerps is None:
            return self.decay
        return min(max(1.0 - (1.0 + self.lerps) ** -self.power, 0.0), self.decay)

    @torch.no_grad()
    def update(self) -> None:  # called only after a successful G update
        if self.lists is None:
            # load_state_dict also resizes per-channel QAT observers
            self.module.load_state_dict(self.net.state_dict())
            self._bind()
            self.lerps = 0
            return
        if self.lerps is not None:
            self.lerps += 1
        dp, sp, db, sb = self.lists
        torch._foreach_lerp_(dp, sp, 1.0 - self.current_decay())
        if db:
            torch._foreach_copy_(db, sb)

    def state_dict(self) -> dict | None:  # None until the first update
        if self.lists is None:
            return None
        sd = self.module.state_dict()
        if self.power is not None and self.lerps is not None:
            sd[self.KEY] = torch.tensor(self.lerps)
        return sd

    def load_state_dict(self, sd: Mapping) -> None:
        lerps = sd.get(self.KEY)
        if lerps is not None:
            sd = copy.copy(sd)  # shallow; keeps the OrderedDict _metadata
            del sd[self.KEY]
        self.module.load_state_dict(sd)  # strict
        self.lerps = None if lerps is None else int(lerps)
        self._bind()  # initialised: the next update lerps, never re-copies


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
