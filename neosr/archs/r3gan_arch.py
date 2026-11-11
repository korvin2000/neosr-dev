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

from neosr.utils.registry import ARCH_REGISTRY

SLOPE = 0.2


class _Conv2d(torch.autograd.Function):
    """F.conv2d whose input gradient is a conv_transpose2d. R1/R2 differentiate that gradient: a
    transposed conv's backward runs the grouped data/weight-gradient kernels, where ATen's conv
    double backward loops over the groups, convolving with grad_output as a full-size kernel, and
    does so even for a constant weight (the FIR)."""

    @staticmethod
    def forward(ctx, x: Tensor, w: Tensor, stride: int, padding: int, groups: int) -> Tensor:
        out = F.conv2d(x, w, None, stride, padding, 1, groups)
        ctx.save_for_backward(x, w)
        ctx.args = stride, padding, groups
        # conv_transpose2d's output_padding that restores x's size (odd sizes under stride 2)
        ctx.extra = tuple(
            i - ((o - 1) * stride - 2 * padding + k)
            for i, o, k in zip(x.shape[2:], out.shape[2:], w.shape[2:], strict=True)
        )
        return out

    @staticmethod
    def backward(ctx, g: Tensor) -> tuple[Tensor | None, ...]:
        x, w = ctx.saved_tensors
        s, p, groups = ctx.args
        gx = gw = None
        # as ATen's conv backward: no autocast, and only the gradients this pass uses (G's step and
        # R1/R2's inner grad take none of D's weights); a compiled graph traces them all
        used = [
            ctx.needs_input_grad[i]
            and (
                torch.compiler.is_compiling()
                or torch._C._will_engine_execute_node(ctx.next_functions[i][0])
            )
            for i in (0, 1)
        ]
        with torch.autocast(g.device.type, enabled=False):
            if used[0]:
                gx = F.conv_transpose2d(g, w, None, s, p, ctx.extra, groups)
            if used[1]:
                gw = torch.ops.aten.convolution_backward(
                    g, x, w, None, [s, s], [p, p], [1, 1], False, [0, 0], groups,
                    [False, True, False],
                )[1]  # fmt: skip
        return gx, gw, None, None, None


def conv2d(x: Tensor, w: Tensor, stride: int = 1, padding: int = 0, groups: int = 1) -> Tensor:
    """F.conv2d (bias-free, square stride/padding) with a cheap double backward; autocast's cast is
    made explicit so the function sees one dtype."""
    dev = x.device.type
    if torch.is_autocast_enabled(dev):  # autocast's rule: every float but fp64 to its dtype
        dt = torch.get_autocast_dtype(dev)
        x, w = (t if t.dtype is torch.float64 else t.to(dt) for t in (x, w))
    # the backward asks the engine about its edges; a leaf's cannot be asked under autograd.grad
    x, w = (t.view_as(t) if t.is_leaf and t.requires_grad else t for t in (x, w))
    return _Conv2d.apply(x, w, stride, padding, groups)


class Conv(nn.Conv2d):
    """A bias-free nn.Conv2d (same parameters and state_dict) through conv2d()."""

    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        return conv2d(x, weight, self.stride[0], self.padding[0], self.groups)


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
    return msr(Conv(cin, cout, k, padding=k // 2, groups=groups, bias=False), gain)


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
        return self.proj(conv2d(x, self.fir, 2, 1, x.shape[1]))


@ARCH_REGISTRY.register(scale=None)
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
        self.basis = msr(Conv(c, c, 4, groups=c, bias=False)) if scalar else None
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


@ARCH_REGISTRY.register(scale=None)
def r3gan_scalar128(**kwargs) -> r3gan:
    """The paper head for GT 128 (six stages)."""
    return r3gan(widths=(32, 64, 128, 256, 256, 256), head="scalar", **kwargs)


@ARCH_REGISTRY.register(scale=None)
def r3gan_scalar256(**kwargs) -> r3gan:
    """The paper head for GT 256 (seven stages)."""
    return r3gan(widths=(32, 64, 128, 256, 256, 256, 256), head="scalar", **kwargs)
