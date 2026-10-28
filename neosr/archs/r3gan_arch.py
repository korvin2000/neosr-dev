"""R3GAN discriminator (Huang et al. 2024, `brownvc/R3GAN` Networks.py block mechanics) for SR.

Pure torch, no normalisation, dropout or buffer writes: every sample is scored independently and the
whole net has a double backward (R1/R2 penalties). Fix-up init makes every block the identity at init.
"""

import math
from collections import OrderedDict
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

SLOPE = 0.2


def msr(conv: nn.Conv2d, gain: float = 1.0) -> nn.Conv2d:
    """The reference's MSRInitializer: weight ~ normal(0, gain / sqrt(fan_in)), no bias."""
    if gain:
        nn.init.normal_(conv.weight, 0.0, gain / math.sqrt(conv.weight[0].numel()))
    else:
        nn.init.zeros_(conv.weight)
    return conv


def conv(
    cin: int, cout: int, k: int = 1, groups: int = 1, gain: float = 1.0
) -> nn.Conv2d:
    return msr(nn.Conv2d(cin, cout, k, padding=k // 2, groups=groups, bias=False), gain)


class BiasedLeaky(nn.Module):
    """bias + leaky_relu(0.2) in place on the fresh sum (saves its output, not its input); forward
    gain 1 like the reference's bias_act: sqrt(2 / (1 + 0.2^2)) is an init gain only."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(c))

    def forward(self, x: Tensor) -> Tensor:
        return F.leaky_relu(
            x + self.bias.to(x.dtype)[:, None, None], SLOPE, inplace=True
        )


class Block(nn.Module):
    """x + lin3(act2(lin2(act1(lin1(x))))): 1x1 expand, grouped 3x3, 1x1 project (zero init)."""

    def __init__(self, c: int, expansion: int, group_width: int, gain: float) -> None:
        super().__init__()
        e = expansion * c
        if e % group_width:
            msg = f"r3gan: expanded width {e} is not a multiple of group_width {group_width}"
            raise ValueError(msg)
        self.lin1, self.act1 = conv(c, e, gain=gain), BiasedLeaky(e)
        self.lin2 = conv(e, e, 3, groups=e // group_width, gain=gain)
        self.act2 = BiasedLeaky(e)
        self.lin3 = conv(e, c, gain=0.0)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.lin3(self.act2(self.lin2(self.act1(self.lin1(x)))))


class Down(nn.Module):
    """Anti-aliased stride 2: the FIR [1,3,3,1] x [1,3,3,1] / 64 as a depthwise conv (keeps NHWC,
    no reshape copy), then a 1x1 projection when the width changes."""

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        k = torch.tensor([1.0, 3.0, 3.0, 1.0])
        fir = (k[:, None] * k[None] / 64).expand(cin, 1, 4, 4).clone()
        self.register_buffer("fir", fir, persistent=False)
        self.proj = conv(cin, cout) if cin != cout else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(F.conv2d(x, self.fir, stride=2, padding=1, groups=x.shape[1]))


class r3gan(nn.Module):
    """R3GAN discriminator; output (B, 1, H', W'), one logit per position.

    Args:
    ----
        widths: channels per stage; every stage but the last halves the resolution.
        group_width: channels per group of the 3x3 conv (the paper's 16).
        expansion: width factor inside a block. blocks: residual blocks per stage.
        head: 'patch' (1x1 conv, stride 2^(len(widths)-1)) or 'scalar' (the paper's 4x4 depthwise
            basis + 1x1, needs an exact 4x4 final map: output (B, 1, 1, 1)).

    """

    def __init__(
        self,
        in_ch: int = 3,
        widths: Sequence[int] = (32, 64, 128, 256, 256),
        group_width: int = 16,
        expansion: int = 2,
        blocks: int = 1,
        head: str = "patch",
    ) -> None:
        super().__init__()
        if head not in {"patch", "scalar"}:
            msg = f"r3gan: head must be 'patch' or 'scalar', got {head!r}"
            raise ValueError(msg)
        n = len(widths)
        # fix-up: the activation's init gain, scaled by (blocks in D)^(-1/(2L-2)) for L = 3 layers
        gain = math.sqrt(2 / (1 + SLOPE**2)) * (blocks * n) ** -0.25
        self.extraction = conv(in_ch, widths[0])
        stages = []
        for i, c in enumerate(widths):
            layers = OrderedDict(
                blocks=nn.Sequential(
                    *(Block(c, expansion, group_width, gain) for _ in range(blocks))
                )
            )
            if i < n - 1:
                layers["down"] = Down(c, widths[i + 1])
            stages.append(nn.Sequential(layers))
        self.stages = nn.Sequential(*stages)
        c = widths[-1]
        scalar = head == "scalar"
        self.basis = msr(nn.Conv2d(c, c, 4, groups=c, bias=False)) if scalar else None
        self.head = conv(c, 1)
        self.gt = 4 * 2 ** (n - 1)  # the input size whose final map is 4x4

    def forward(self, x: Tensor) -> Tensor:
        x = self.stages(self.extraction(x))
        if self.basis is not None:
            if x.shape[-2:] != (4, 4):
                msg = (
                    f"r3gan_scalar head: the final map is {x.shape[-2]}x{x.shape[-1]}, needs 4x4: "
                    f"train with GT {self.gt} or use r3gan"
                )
                raise ValueError(msg)
            x = self.basis(x)
        return self.head(x)


def r3gan_scalar128(**kwargs) -> r3gan:
    """The paper head for GT 128 (six stages)."""
    return r3gan(widths=(32, 64, 128, 256, 256, 256), head="scalar", **kwargs)


def r3gan_scalar256(**kwargs) -> r3gan:
    """The paper head for GT 256 (seven stages)."""
    return r3gan(widths=(32, 64, 128, 256, 256, 256, 256), head="scalar", **kwargs)
