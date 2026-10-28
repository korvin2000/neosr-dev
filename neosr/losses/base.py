"""Loss contract. A loss reads what it needs from Context; trainers only call compute(ctx)."""

from contextlib import nullcontext
from functools import cached_property

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _RestoringClamp(torch.autograd.Function):
    """clamp(x, lo, hi) whose backward keeps the gradient components that move an out-of-range raw
    value back into range (T-01). The hard clamp zeroes every out-of-range pixel's gradient, so such
    a pixel can never return; a straight-through clamp would let an outward gradient drift it."""

    @staticmethod
    def forward(ctx, x: Tensor, lo: float, hi: float) -> Tensor:
        ctx.save_for_backward(x)
        ctx.lo, ctx.hi = lo, hi
        return x.clamp(lo, hi)

    @staticmethod
    def backward(ctx, g: Tensor) -> tuple[Tensor, None, None]:
        (x,) = ctx.saved_tensors
        out = ((x < ctx.lo) & (g > 0)) | ((x > ctx.hi) & (g < 0))
        return g.masked_fill(out, 0), None, None


def restoring_clamp(x: Tensor, lo: float = 0.0, hi: float = 1.0) -> Tensor:
    return _RestoringClamp.apply(x, lo, hi)


class Context:
    """What one generator forward exposes to the losses. Built by the trainer; losses only read it."""

    def __init__(
        self,
        *,
        sr: Tensor,
        gt: Tensor,
        lq: Tensor,
        it: int,
        net_d: nn.Module | None = None,
        net_d_adv: nn.Module | None = None,
    ) -> None:
        self.sr, self.gt, self.lq, self.it = sr, gt, lq, it
        self.net_d = net_d  # D execution wrapper (DDP/compile): D's own update
        self.net_d_adv = net_d_adv  # D without a DDP reducer: G's adversarial pass
        self.logs: dict[str, Tensor] = {}

    @cached_property
    def lq_up(self) -> Tensor:
        """Colour reference: bicubic antialiased lq at sr size, clamped to [1/255, 1] (was match_lq_colors)."""
        with torch.no_grad(), torch.autocast(self.lq.device.type, enabled=False):
            up = F.interpolate(
                self.lq.float(), size=self.sr.shape[-2:], mode="bicubic", antialias=True
            )
        return up.clamp_(1 / 255, 1)

    def log(self, key: str, value: Tensor) -> None:
        self.logs[key] = value.detach()


class Loss(nn.Module):
    """forward(*tensors) is the unweighted kernel. evaluate(ctx) picks inputs/models. compute(ctx) is final."""

    loss_weight: float = 1.0  # class default; the builder sets the instance value
    target: str = "gt"  # Context attribute used as the reference: "gt" | "lq_up"
    phase: str = "g"  # objective this term joins: "g" | "d"
    log_key: str = ""  # set by the builder; "" = not logged on its own
    fp32: bool = False  # run outside autocast with fp32 inputs (was custom_fwd(cast_inputs=float32))

    def expand(self) -> list["Loss"]:
        return [self]

    def active(self, it: int) -> bool:  # noqa: ARG002
        """Whether this term joins the objective at update `it` (1-based); a phase without an
        active term idles for that update (no forward through its nets, no step, no skip)."""
        return True

    def evaluate(self, ctx: Context) -> Tensor:
        sr, ref = ctx.sr, getattr(ctx, self.target)
        return self(sr.float(), ref.float()) if self.fp32 else self(sr, ref)

    def compute(self, ctx: Context) -> Tensor:
        fp32 = torch.autocast(ctx.sr.device.type, enabled=False)
        with fp32 if self.fp32 else nullcontext():
            value = self.evaluate(ctx) * self.loss_weight  # the only weighting
        if value.ndim:
            msg = f"{type(self).__name__} returned shape {tuple(value.shape)}, expected a scalar"
            raise ValueError(msg)
        if self.log_key:
            ctx.log(self.log_key, value)
        return value
