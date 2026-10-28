"""Training-step building blocks: one Phase per optimizer, the generator EMA and the log meter."""

from __future__ import annotations

import copy
import logging
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
    """adamw_sf is always SF; adan_sf honours its (effective, per-group) flag; plain optimizers have
    no eval()."""
    return callable(getattr(opt, "eval", None)) and any(
        g.get("schedule_free", True) for g in opt.param_groups
    )


# grad_clip = "spike": clip only above SPIKE_K x an EMA (horizon 1 / (1 - SPIKE_BETA) = 50 updates)
# of the applied, post-clip norms (T-02): normal steps stay untouched, a x50 spike is cut to 3x
SPIKE_K, SPIKE_BETA = 3.0, 0.98
# param-group entries that are optimizer state, not configuration (SF averaging, step counters);
# `lr` is compared as configured: `initial_lr` when a scheduler owns it
_GROUP_STATE = frozenset({
    "params", "lr", "initial_lr", "step", "k", "train_mode", "weight_sum", "lr_max",
})  # fmt: skip


def _hparams(group: Mapping) -> dict:
    out = {
        k: tuple(v) if isinstance(v, list) else v
        for k, v in group.items()
        if k not in _GROUP_STATE and isinstance(v, int | float | tuple | list)
    }
    out["lr"] = group.get("initial_lr", group["lr"])
    return out


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
        grad_clip: float | str,
        max_skip_streak: int = 10,
        schedulers: Iterable = (),
        names: Sequence[str] | None = None,
        opt_type: str | None = None,
        strict_unused: bool = False,
    ) -> None:
        self.name, self.params, self.opt, self.terms = name, params, opt, terms
        # resume identity: config optimizer type, its state version, SF mode, ordered trainable
        # names; no positional fallback: same-count swaps would silently remap optimizer moments
        self.opt_type, self.sf = opt_type or type(opt).__name__, is_schedule_free(opt)
        self.version = getattr(type(opt), "STATE_VERSION", 0)
        self.names = None if names is None else list(names)
        if self.names is not None and len(self.names) != len(params):
            msg = f"net_{name}: {len(self.names)} names for {len(params)} parameters"
            raise ValueError(msg)
        self.autocast = {"device_type": "cuda", "dtype": amp_dtype, "enabled": use_amp}
        # fp16: a safe start (2**5, no skip streak at init) that reaches the usual 2**16 in ~2k
        # updates instead of ~22k (ST-12); a growth probe costs one isolated skip per 200 updates
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=use_amp and amp_dtype is torch.float16,
            init_scale=2.0**5,
            growth_interval=200,
        )
        self.inv_accum = 1.0 / accumulate
        # a fixed max-norm (true = 1.0, false = 0.0 = off) or "spike"
        self.spike = grad_clip == "spike"
        self.grad_clip = 0.0 if self.spike else float(grad_clip)
        # "spike": None until the first applied update; while it is 0 (zero gradients) nothing is
        # clipped and the next applied norm re-seeds it, so a zero start cannot freeze training
        self.norm_ema: Tensor | None = None
        self.max_skip_streak, self.schedulers = max_skip_streak, list(schedulers)
        # host ints: skipped in the log window, consecutive skips
        self.skipped = self.streak = 0
        # device scalars of the last applied update: pre-clip norm, 1.0 if clipped (None: no clip)
        self.grad_norm: Tensor | None = None
        self.clipped: Tensor | None = None
        # no active term in this window (Loss.active): no backward, no step, no skip
        self.idle = False
        # T-10: list parameters without a gradient once; DDP without find_unused_parameters raises
        self.check_unused, self.strict_unused = True, strict_unused

    def active(self, it: int) -> bool:
        """Any term active in update `it`: a pure function of it, so every micro-batch and rank agree."""
        return any(t.active(it) for t in self.terms)

    def backward(self, ctx: Context) -> None:
        terms = [t for t in self.terms if t.active(ctx.it)]
        self.idle = not terms
        if self.idle:
            return
        with torch.autocast(**self.autocast):
            total = sum(t.compute(ctx) for t in terms)
        ctx.log(f"l_{self.name}_total", total)  # undivided micro-batch objective
        self.scaler.scale(total * self.inv_accum).backward(inputs=self.params)

    def step(self) -> bool | None:
        """True: applied; False: skipped (non-finite gradient norm); None: idle window."""
        if self.idle:  # the schedule clock still counts the update
            self.opt.zero_grad(set_to_none=True)
            for s in self.schedulers:
                s.step()
            return None
        if self.check_unused:
            self._check_unused()
        self.scaler.unscale_(self.opt)  # no-op unless fp16
        grads = [p.grad for p in self.params if p.grad is not None]
        norm = get_total_norm(grads, foreach=True)
        thr: float | Tensor | None = self.grad_clip or None
        if thr is not None:
            clip_grads_with_norm_(self.params, thr, norm, foreach=True)
        elif self.norm_ema is not None:  # "spike"; the first applied update seeds it
            thr = torch.where(self.norm_ema > 0, self.norm_ema * SPIKE_K, torch.inf)
            # clip_grads_with_norm_'s formula; it would float() a tensor threshold (a host sync)
            torch._foreach_mul_(grads, (thr / (norm + 1e-6)).clamp_(max=1.0))
        # the one host sync per update; rank-identical under DDP
        ok = bool(norm.isfinite())
        if ok:
            # fp16: re-checks found_inf on the drained stream
            self.scaler.step(self.opt)
            self.grad_norm = norm
            self.clipped = None if thr is None else (norm > thr).float()
            if self.spike:  # EMA of the norms as applied (post-clip)
                n = norm if thr is None else torch.minimum(norm, thr)
                e = self.norm_ema
                self.norm_ema = (
                    n.clone()
                    if e is None
                    else torch.where(e > 0, e.lerp(n, 1 - SPIKE_BETA), n)
                )
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

    def _check_unused(self) -> None:
        """First non-idle boundary: trainable parameters without a gradient (host attribute, no sync).
        DDP's reducer would fail at the next forward unless find_unused_parameters is set."""
        self.check_unused = False
        names = self.names or [f"#{i}" for i in range(len(self.params))]
        dead = [n for n, p in zip(names, self.params, strict=True) if p.grad is None]
        if not dead:
            return
        msg = (
            f"net_{self.name}: {len(dead)} of {len(self.params)} trainable parameters got no "
            f"gradient (first: {', '.join(dead[:5])})"
        )
        if self.strict_unused:
            msg += "; under DDP set find_unused_parameters = true (or fix the network)"
            raise ConfigError(msg)
        logging.getLogger("neosr").warning(
            f"{msg}: they are never updated or weight-decayed"
        )

    def _need_names(self) -> list[str]:
        if self.names is None:
            msg = f"net_{self.name}: state needs the trainable names (pass names=)"
            raise RuntimeError(msg)
        return self.names

    def state_dict(self) -> dict:
        return {
            "optimizer": {
                "type": self.opt_type,
                "version": self.version,
                "sf": self.sf,
                "names": self._need_names(),
                "state": self.opt.state_dict(),
            },
            "scaler": self.scaler.state_dict(),  # {} when fp16 is off
            "schedulers": [
                {"type": type(s).__name__, "state": s.state_dict()}
                for s in self.schedulers
            ],
            "clip_ema": self.norm_ema,  # "spike" only
        }

    def load_state_dict(self, sd: Mapping, log: Logger) -> None:
        """Strict resume: a changed optimizer, state version, SF mode, trainable set or scheduler
        list raises. Saved hyper-parameters win over the config, with a warning per change."""
        o, hint = sd["optimizer"], "; start a new run with pretrain_network_*"
        if (o["type"], o["sf"]) != (self.opt_type, self.sf):
            msg = (
                f"net_{self.name}: optimizer {o['type']}(sf={o['sf']}) -> "
                f"{self.opt_type}(sf={self.sf}){hint}"
            )
            raise ConfigError(msg)
        if (
            v := o.get("version", 0)
        ) != self.version:  # absent: saved before versioning
            msg = (
                f"net_{self.name}: {self.opt_type} state version {v} -> {self.version} "
                f"(the update rule changed){hint}"
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
        groups = o["state"]["param_groups"]
        for i, (old, new) in enumerate(
            zip(groups, self.opt.param_groups, strict=False)
        ):
            now, where = _hparams(new), f" (group {i})" if len(groups) > 1 else ""
            for k, v in _hparams(old).items():
                if k in now and v != now[k]:
                    log.warning(
                        f"net_{self.name}{where}: restored {k} {v!r} from the state (config "
                        f"{now[k]!r}); use pretrain_network_{self.name} to change it"
                    )
        self.opt.load_state_dict(o["state"])  # restores SF train_mode group flags
        for s, ss in zip(self.schedulers, sd["schedulers"], strict=True):
            s.load_state_dict(ss["state"])
        if self.spike and (ema := sd.get("clip_ema")) is not None:
            self.norm_ema = ema.to(self.params[0].device)
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
    """Window means of detached device scalars: masked device sums and finite counts, no host sync
    until read() (one cat + all_reduce + tolist). A micro-batch's values wait for its update
    (`commit`), tagged with the phase whose step decides about them. A non-finite value is left out
    of the mean and counted as `nonfinite_<key>`; in a skipped step that is all (ST-02: the step was
    already refused). In an applied step the gradients were finite, so the objective itself is
    broken: read() raises (T1 D15)."""

    def __init__(self) -> None:
        self.pending: list[tuple[str | None, Mapping[str, Tensor]]] = []
        # (keys, applied) -> [device sums, device finite counts, host number of adds]
        self.groups: dict[tuple[tuple[str, ...], bool], list] = {}

    def add(self, logs: Mapping[str, Tensor], phase: str | None = None) -> None:
        if logs:
            self.pending.append((phase, logs))

    def commit(self, applied: bool | Mapping[str | None, bool]) -> None:
        """At an update boundary: was the step of each value's phase applied (a bool: of all)?"""
        for phase, logs in self.pending:
            a = applied if isinstance(applied, bool) else applied.get(phase, True)
            vec = torch.stack([v.float() for v in logs.values()])
            ok = vec.isfinite()
            val = torch.where(ok, vec, 0.0)
            g = self.groups.get(key := (tuple(logs), a))
            if g is None:
                self.groups[key] = [val, ok.float(), 1]
            else:
                g[0].add_(val)
                g[1].add_(ok)
                g[2] += 1
        self.pending.clear()

    def read(self, reset: bool = True) -> dict[str, float]:
        self.commit(applied=True)  # direct use without commit: strict
        if not self.groups:
            return {}
        vec = torch.cat([t for g in self.groups.values() for t in g[:2]])
        world = 1
        if dist.is_initialized():
            dist.all_reduce(vec)  # SUM; collective: call on all ranks
            world = dist.get_world_size()
        flat, i = vec.tolist(), 0
        tot: dict[
            str, list[float]
        ] = {}  # key -> [sum, finite, non-finite, non-finite applied]
        for (keys, applied), (*_, n) in self.groups.items():
            k = len(keys)
            for key, s, c in zip(
                keys, flat[i : i + k], flat[i + k : i + 2 * k], strict=True
            ):
                t = tot.setdefault(key, [0.0, 0.0, 0.0, 0.0])
                bad = n * world - c
                t[0], t[1], t[2] = t[0] + s, t[1] + c, t[2] + bad
                t[3] += bad if applied else 0.0
            i += 2 * k
        if reset:
            self.groups.clear()
        out = {k: s / c for k, (s, c, _, _) in tot.items() if c}
        out |= {f"nonfinite_{k}": b for k, (_, _, b, _) in tot.items() if b}
        if broken := [k for k, t in tot.items() if t[3]]:
            raise FloatingPointError(
                f"non-finite training values {broken} in applied updates (finite gradients): "
                "a broken loss; nothing was saved from this window"
            )
        return out
