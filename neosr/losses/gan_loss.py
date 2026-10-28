import math
from contextlib import nullcontext

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.modules import batchnorm

from neosr.config import ConfigError
from neosr.losses.base import Context, Loss
from neosr.losses.wavelet_guided import AdvView


def batch_coupled(net: nn.Module) -> bool:
    """D normalises with batch statistics: real and fake then need separate forwards."""
    return any(isinstance(m, batchnorm._BatchNorm) for m in net.modules())


def eager(net: nn.Module) -> nn.Module:
    """The module under torch.compile (a compiled graph has no double backward)."""
    return getattr(net, "_orig_mod", net)


def signs(o: Tensor) -> Tensor:
    """Fresh Rademacher +-1 of o's shape from the device RNG (saved with the training state)."""
    return torch.empty_like(o).bernoulli_(0.5).mul_(2).sub_(1)


def probe(out: Tensor) -> Tensor:
    """The score whose input gradient gives the R1/R2 norm: with fresh Rademacher signs eps per
    sample and position, E||grad_x sum_p eps_p D_p(x) / sqrt(P)||^2 = mean_p ||grad_x D_p(x)||^2
    (the cross terms vanish), i.e. every position of a map D is a critic, at one backward's cost.
    P = 1 (a scalar D) draws nothing: the plain per-sample score."""
    o = out.float().reshape(len(out), -1)
    if (p := o.shape[1]) == 1:
        return o.sum()
    return (o * signs(o)).sum() / math.sqrt(p)


def _number(v: object) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def _count(v: object, low: int) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= low


class gan_loss(Loss):
    """[train.gan_opt]: the generator's adversarial term; expand() adds the discriminator's term.

    Args:
    ----
        gan_type (str): 'bce', 'mse' (l2), 'huber', 'hinge' (D: relu(1 - real) + relu(1 + fake),
            G: -fake; no labels) or 'rpgan' (relativistic pairing, R3GAN's RpGAN: D scores each fake
            against its own real; no labels).
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        wavelet_init (int | None): D sees the SWT high-frequency view from this update on;
            None: D always sees RGB.
        start_iter / ramp_iters / end_iter: the generic schedule of every term (Loss.schedule);
            D trains exactly while the G term is active, and D's term is never ramped.
        r1_gamma, r2_gamma (float): R1 (real) / R2 (fake) penalty gamma / 2 * ||grad_x D(x)||^2
            on D's input, per sample; a map D: the mean over positions (each position is a critic,
            Rademacher estimate). Default: 0 (off).
        reg_every (int): the penalty runs on every reg_every-th update, scaled by reg_every
            (lazy; R3GAN: 1). Default: 16.
        gamma_decay, gamma_iters: both gammas follow a cosine from 1x to gamma_decay x over
            gamma_iters updates counted from the first GAN update (R3GAN: 0.1). Default: 1, 0
            (constant).
        penalty_fp32 (bool): run the penalty pass in fp32 also when the phase is bf16 (parity
            checks); fp16 and fp32 phases always run it in fp32. Default: False.

    """

    loss_weight = 0.1  # the generator term only; the discriminator term is unweighted

    def __init__(
        self,
        gan_type: str = "bce",
        real_label_val: float = 1.0,
        fake_label_val: float = 0.0,
        wavelet_init: int | None = None,
        r1_gamma: float = 0.0,
        r2_gamma: float = 0.0,
        reg_every: int = 16,
        gamma_decay: float = 1.0,
        gamma_iters: int = 0,
        penalty_fp32: bool = False,
    ) -> None:
        super().__init__()
        crits = {"bce": nn.BCEWithLogitsLoss, "mse": nn.MSELoss, "huber": nn.HuberLoss}
        if gan_type not in {*crits, "hinge", "rpgan"}:
            msg = f"gan_type must be one of {sorted({*crits, 'hinge', 'rpgan'})}, got {gan_type!r}"
            raise ValueError(msg)
        if gan_type not in crits and (real_label_val, fake_label_val) != (1.0, 0.0):
            msg = f"gan_type {gan_type!r} takes no labels: leave real_label_val/fake_label_val unset"
            raise ValueError(msg)
        gammas = (r1_gamma, r2_gamma)
        if not all(_number(gm) and gm >= 0 for gm in gammas):
            msg = f"gan_loss: r1_gamma and r2_gamma must be finite numbers >= 0, got {gammas}"
            raise ValueError(msg)
        if not (_count(reg_every, 1) and _count(gamma_iters, 0)):
            msg = (
                f"gan_loss: reg_every must be an integer >= 1 and gamma_iters an integer >= 0, "
                f"got {reg_every!r}, {gamma_iters!r}"
            )
            raise ValueError(msg)
        if not (_number(gamma_decay) and 0 <= gamma_decay <= 1):
            msg = f"gan_loss: gamma_decay must be in [0, 1], got {gamma_decay!r}"
            raise ValueError(msg)
        if gamma_iters and not any(gammas):
            msg = (
                "gan_loss: gamma_iters schedules a penalty, but r1_gamma = r2_gamma = 0"
            )
            raise ValueError(msg)
        if gamma_decay != 1 and not gamma_iters:
            msg = "gan_loss: gamma_decay needs gamma_iters > 0 (the cosine length)"
            raise ValueError(msg)
        if not isinstance(penalty_fp32, bool):
            msg = f"gan_loss: penalty_fp32 must be true or false, got {penalty_fp32!r}"
            raise ValueError(msg)  # ruff: ignore[type-check-without-type-error] (a config value)
        self.rp, self.hinge = gan_type == "rpgan", gan_type == "hinge"
        self.crit = crits[gan_type]() if gan_type in crits else None
        self.labels = (fake_label_val, real_label_val)
        self.view = None if wavelet_init is None else AdvView(wavelet_init)
        self.gammas, self.reg_every = gammas, reg_every
        self.gamma_decay, self.gamma_iters = gamma_decay, gamma_iters
        self.penalty_fp32 = penalty_fp32

    def gamma_scale(self, it: int) -> float:
        """The gammas' factor at update `it`: 1 at the first GAN update (start_iter + 1), a cosine
        to gamma_decay over gamma_iters updates, then constant. A pure host function of `it`."""
        if not (n := self.gamma_iters):
            return 1.0
        t, d = min(max(it - self.start_iter - 1, 0), n), self.gamma_decay
        return d + (1 - d) * (1 + math.cos(math.pi * t / n)) / 2

    def adv(self, x: Tensor, it: int) -> Tensor:
        return x if self.view is None else self.view(x, it)

    def term(self, logits: Tensor, real: bool) -> Tensor:
        return self.crit(logits, torch.full_like(logits, self.labels[real]))  # type: ignore[reportOptionalCall]

    def evaluate(self, ctx: Context) -> Tensor:
        # train-mode D without a DDP reducer; the trainer's backward(inputs=G params) keeps D's grads out
        d, it = ctx.net_d_adv, ctx.it
        if self.rp:
            with torch.no_grad():  # the pair's real side is a constant for G
                real = d(self.adv(ctx.gt, it))  # type: ignore[reportOptionalCall]
            fake = d(self.adv(ctx.sr, it))  # type: ignore[reportOptionalCall]
            return F.softplus(real.float() - fake.float()).mean()
        fake = d(self.adv(ctx.sr, it))  # type: ignore[reportOptionalCall]
        return -fake.float().mean() if self.hinge else self.term(fake, real=True)

    def expand(self) -> list[Loss]:
        return [self, gan_d_loss(self)]


class gan_d_loss(Loss):
    """The discriminator objective: one fresh forward on cat([view(gt), view(sr.detach())]) (two
    for a D with batch statistics, which cannot take R1/R2), plus R1/R2 on regularised updates."""

    phase = "d"

    def __init__(self, g: gan_loss) -> None:
        super().__init__()
        self.g = g
        self.split: bool | None = None  # batch_coupled(D), decided at the first call

    def active(self, it: int) -> bool:
        return self.g.active(it)  # D trains exactly while G's adversarial term is on

    def scores(
        self, net: nn.Module, real: Tensor, fake: Tensor
    ) -> tuple[Tensor, Tensor]:
        if self.split is None:
            self.split = batch_coupled(net)
            if self.split and any(self.g.gammas):
                msg = (
                    "[train.gan_opt] r1_gamma/r2_gamma need a D that scores every sample on "
                    "its own; network_d normalises with batch statistics (BatchNorm)"
                )
                raise ConfigError(msg)
        if self.split:
            return net(real), net(fake)
        return net(torch.cat([real, fake])).chunk(2)  # type: ignore[return-value]

    def evaluate(self, ctx: Context) -> Tensor:
        g, it = self.g, ctx.it
        real_in, fake_in = g.adv(ctx.gt, it), g.adv(ctx.sr.detach(), it)
        # the effective gammas, host floats: ranks agree, no sync; zero skips the graph and the draw
        m = 0.0 if it % g.reg_every else g.gamma_scale(it)
        gammas = [gm * m for gm in g.gammas]
        if not any(gammas):
            real, fake = self.scores(ctx.net_d, real_in, fake_in)
            return self.adv_loss(real, fake, ctx)
        # R1/R2 through the eager D (compiled graphs and flash/mem-efficient SDPA have no double
        # backward) on fp32 leaves; one forward serves the loss and the penalty. A bf16 phase keeps
        # bf16 (norms accumulate in fp32); fp16 runs fp32: its inner autograd.grad is not loss-scaled
        dev = ctx.sr.device.type
        bf16 = (
            not g.penalty_fp32
            and torch.is_autocast_enabled(dev)
            and torch.get_autocast_dtype(dev) is torch.bfloat16
        )
        xs = [
            x.detach().float().requires_grad_(gm > 0)
            for x, gm in zip((real_in, fake_in), gammas, strict=True)
        ]
        with (
            nullcontext() if bf16 else torch.autocast(dev, enabled=False),
            sdpa_kernel(SDPBackend.MATH),
        ):
            real, fake = self.scores(eager(ctx.net_d), *xs)
            on = [
                (out, x, gm, side)
                for out, x, gm, side in zip(
                    (real, fake), xs, gammas, ("r1", "r2"), strict=True
                )
                if gm > 0
            ]
            score = sum(probe(out) for out, *_ in on)  # real side draws first
            grads = torch.autograd.grad(
                score, [x for _, x, *_ in on], create_graph=True
            )
        raws = [gx.float().square().flatten(1).sum(1).mean() for gx in grads]
        # the old arithmetic order: P = 1 stays bitwise the pre-R3GAN penalty for any reg_every
        reg = (
            0.5
            * g.reg_every
            * sum(gm * r for (*_, gm, _), r in zip(on, raws, strict=True))
        )
        for (*_, gm, side), raw in zip(on, raws, strict=True):
            ctx.log(side, raw)  # E||grad_x D||^2 before gamma (per position for maps)
            ctx.log(f"l_d_{side}", 0.5 * g.reg_every * gm * raw)
            ctx.log(f"gamma_{side}", torch.full((), gm, device=raw.device))
        ctx.log("l_d_reg", reg)
        return self.adv_loss(real, fake, ctx) + reg

    def adv_loss(self, real: Tensor, fake: Tensor, ctx: Context) -> Tensor:
        if self.g.rp:
            value = F.softplus(fake.float() - real.float()).mean()
            logs = {"out_d_margin": (real.float() - fake.float()).mean()}
        else:
            if self.g.hinge:
                l_real = F.relu(1 - real.float()).mean()
                l_fake = F.relu(1 + fake.float()).mean()
            else:
                l_real, l_fake = (
                    self.g.term(real, real=True),
                    self.g.term(fake, real=False),
                )
            value, logs = l_real + l_fake, {"l_d_real": l_real, "l_d_fake": l_fake}
        logs |= {"out_d_real": real.mean(), "out_d_fake": fake.mean()}
        for key, v in logs.items():
            ctx.log(key, v)
        return value
