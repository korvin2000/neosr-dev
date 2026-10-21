"""Loss contract. A loss reads what it needs from Context; trainers only call compute(ctx)."""

from contextlib import nullcontext
from functools import cached_property

import torch
from torch import Tensor, nn
from torch.nn import functional as F


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
