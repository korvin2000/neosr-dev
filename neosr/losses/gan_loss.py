import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.modules import batchnorm

from neosr.losses.base import Context, Loss
from neosr.losses.wavelet_guided import AdvView


def batch_coupled(net: nn.Module) -> bool:
    """D normalises with batch statistics: real and fake then need separate forwards."""
    return any(isinstance(m, batchnorm._BatchNorm) for m in net.modules())


def eager(net: nn.Module) -> nn.Module:
    """The module under torch.compile (a compiled graph has no double backward)."""
    return getattr(net, "_orig_mod", net)


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
        r1_gamma, r2_gamma (float): lazy R1 (real) / R2 (fake) penalty
            gamma / 2 * ||grad_x D(x)||^2 on D's input, per sample. Default: 0 (off).
        reg_every (int): the penalty runs on every reg_every-th update, scaled by reg_every.
            Default: 16.

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
    ) -> None:
        super().__init__()
        crits = {"bce": nn.BCEWithLogitsLoss, "mse": nn.MSELoss, "huber": nn.HuberLoss}
        if gan_type not in {*crits, "hinge", "rpgan"}:
            msg = f"gan_type must be one of {sorted({*crits, 'hinge', 'rpgan'})}, got {gan_type!r}"
            raise ValueError(msg)
        if gan_type not in crits and (real_label_val, fake_label_val) != (1.0, 0.0):
            msg = f"gan_type {gan_type!r} takes no labels: leave real_label_val/fake_label_val unset"
            raise ValueError(msg)
        if min(r1_gamma, r2_gamma) < 0 or reg_every < 1:
            msg = "gan_loss: r1_gamma and r2_gamma must be >= 0, reg_every >= 1"
            raise ValueError(msg)
        self.rp, self.hinge = gan_type == "rpgan", gan_type == "hinge"
        self.crit = crits[gan_type]() if gan_type in crits else None
        self.labels = (fake_label_val, real_label_val)
        self.view = None if wavelet_init is None else AdvView(wavelet_init)
        self.gammas, self.reg_every = (r1_gamma, r2_gamma), reg_every

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
    for a D with batch statistics), plus the lazy R1/R2 penalty on regularised updates."""

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
        if self.split:
            return net(real), net(fake)
        return net(torch.cat([real, fake])).chunk(2)  # type: ignore[return-value]

    def evaluate(self, ctx: Context) -> Tensor:
        g, it = self.g, ctx.it
        real_in, fake_in = g.adv(ctx.gt, it), g.adv(ctx.sr.detach(), it)
        if not any(g.gammas) or it % g.reg_every:  # host ints: ranks agree, no sync
            real, fake = self.scores(ctx.net_d, real_in, fake_in)
            return self.adv_loss(real, fake, ctx)
        # lazy R1/R2: an fp32 eager D (flash/mem-efficient SDPA and compiled graphs have no
        # double backward) on inputs that require grad; one forward serves the loss and the penalty
        xs = [
            x.detach().float().requires_grad_(gm > 0)
            for x, gm in zip((real_in, fake_in), g.gammas, strict=True)
        ]
        with (
            torch.autocast(ctx.sr.device.type, enabled=False),
            sdpa_kernel(SDPBackend.MATH),
        ):
            real, fake = self.scores(eager(ctx.net_d), *xs)
            on = [
                (out, x, gm)
                for out, x, gm in zip((real, fake), xs, g.gammas, strict=True)
                if gm > 0
            ]
            score = sum(out.flatten(1).mean(1).sum() for out, _, _ in on)  # per sample
            grads = torch.autograd.grad(score, [x for _, x, _ in on], create_graph=True)
            pen = sum(
                gm * gx.square().flatten(1).sum(1).mean()
                for (_, _, gm), gx in zip(on, grads, strict=True)
            )
        reg = 0.5 * g.reg_every * pen
        value = self.adv_loss(real, fake, ctx) + reg
        ctx.log("l_d_reg", reg)
        return value

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
