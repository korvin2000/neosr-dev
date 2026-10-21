import torch
from torch import Tensor, nn

from neosr.losses.base import Context, Loss
from neosr.losses.wavelet_guided import AdvView


class gan_loss(Loss):
    """[train.gan_opt]: the generator's adversarial term; expand() adds the discriminator's term.

    Args:
    ----
        gan_type (str): 'bce', 'mse' (l2) or 'huber'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        wavelet_init (int | None): D sees the SWT high-frequency view from this update on;
            None: D always sees RGB.

    """

    loss_weight = 0.1  # the generator term only; the discriminator term is unweighted

    def __init__(
        self,
        gan_type: str = "bce",
        real_label_val: float = 1.0,
        fake_label_val: float = 0.0,
        wavelet_init: int | None = None,
    ) -> None:
        super().__init__()
        crits = {"bce": nn.BCEWithLogitsLoss, "mse": nn.MSELoss, "huber": nn.HuberLoss}
        if gan_type not in crits:
            msg = f"gan_type must be one of {sorted(crits)}, got {gan_type!r}"
            raise ValueError(msg)
        self.crit = crits[gan_type]()
        self.labels = (fake_label_val, real_label_val)
        self.view = None if wavelet_init is None else AdvView(wavelet_init)

    def adv(self, x: Tensor, it: int) -> Tensor:
        return x if self.view is None else self.view(x, it)

    def term(self, logits: Tensor, real: bool) -> Tensor:
        return self.crit(logits, torch.full_like(logits, self.labels[real]))

    def evaluate(self, ctx: Context) -> Tensor:
        # train-mode D without a DDP reducer; the trainer's backward(inputs=G params) keeps D's grads out
        return self.term(ctx.net_d_adv(self.adv(ctx.sr, ctx.it)), real=True)  # type: ignore[reportOptionalCall]

    def expand(self) -> list[Loss]:
        return [self, gan_d_loss(self)]


class gan_d_loss(Loss):
    """The discriminator objective: fresh forwards on view(gt) and view(sr.detach())."""

    phase = "d"

    def __init__(self, g: gan_loss) -> None:
        super().__init__()
        self.g = g

    def evaluate(self, ctx: Context) -> Tensor:
        real = ctx.net_d(self.g.adv(ctx.gt, ctx.it))  # type: ignore[reportOptionalCall]
        fake = ctx.net_d(self.g.adv(ctx.sr.detach(), ctx.it))  # type: ignore[reportOptionalCall]
        l_real, l_fake = self.g.term(real, real=True), self.g.term(fake, real=False)
        for key, value in (
            ("l_d_real", l_real),
            ("l_d_fake", l_fake),
            ("out_d_real", real.mean()),
            ("out_d_fake", fake.mean()),
        ):
            ctx.log(key, value)
        return l_real + l_fake
