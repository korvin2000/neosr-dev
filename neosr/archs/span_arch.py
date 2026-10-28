from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F


def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value


def conv_layer(in_channels, out_channels, kernel_size, bias=True):
    """Re-write convolution layer for adaptive `padding`."""
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2), int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def sequential(*args):
    """Modules will be added to the a Sequential Container in the order they
    are passed.

    Parameters
    ----------
    args: Definition of Modules in order.
    -------

    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            msg = "sequential does not support OrderedDict input."
            raise NotImplementedError(msg)
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            modules.extend(module.children())
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3):
    """Upsample features according to `upscale_factor`."""
    conv = conv_layer(in_channels, out_channels * (upscale_factor**2), kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


def param_key(m: nn.Module) -> tuple | None:
    """Changes on every in-place parameter update (optimizer step, EMA lerp, load_state_dict) and on
    reallocation (.to(), deepcopy); not on writes through `.data` (eval() refreshes after those).
    None for inference tensors (built under inference_mode: no version counter), which re-fuse on
    every eval forward."""
    ps = list(m.parameters())
    if any(p.is_inference() for p in ps):
        return None
    return tuple((p.data_ptr(), p._version) for p in ps)


def fuse_online(m: nn.Module) -> bool:
    """Run the differentiable fused kernel instead of the cached deploy conv: in training, under
    torch.compile, and in an eval forward that autograd records for trainable branch parameters
    (the deploy conv is frozen, so it would drop their gradients)."""
    return (
        m.training
        or torch.compiler.is_compiling()
        or (
            torch.is_grad_enabled()
            and not torch.onnx.is_in_onnx_export()
            and any(p.requires_grad for p in m.parameters())
        )
    )


def cache_stale(m: nn.Module) -> bool:
    key = param_key(m)
    return key is None or key != m._key


def fuse_conv3xc(sk: nn.Conv2d, conv: nn.Sequential):
    """Fold the 1x1 skip and the 1x1-3x3-1x1 branch into one 3x3 kernel (fp32 under autocast)."""
    with torch.autocast(sk.weight.device.type, enabled=False):
        w1, w2, w3 = (c.weight for c in conv)
        w = (
            F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        w = (
            F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        w = w + F.pad(sk.weight, (1, 1, 1, 1))
        if sk.bias is None:
            return w, None
        b1, b2, b3 = (c.bias for c in conv)
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2
        b = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3
        return w, b + sk.bias


class Conv3XC(nn.Module):
    def __init__(self, c_in, c_out, gain1=1, s=1, bias=True):
        super().__init__()
        self.bias = bias
        self.stride = s
        gain = gain1

        self.sk = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=1,
            padding=0,
            stride=s,
            bias=bias,
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=c_in,
                out_channels=c_in * gain,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_in * gain,
                out_channels=c_out * gain,
                kernel_size=3,
                stride=s,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_out * gain,
                out_channels=c_out,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
        )

        # deploy conv (eval without autograd, export): fused kernel, refreshed on eval(), state_dict()
        # and in an eval forward after any in-place parameter update (EMA lerp, load_state_dict)
        self.eval_conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=3,
            padding=1,
            stride=s,
            bias=bias,
        ).requires_grad_(requires_grad=False)
        self._key = None  # param_key() the eval_conv was fused at

    def fused_kernel(self):
        return fuse_conv3xc(self.sk, self.conv)

    @torch.no_grad()
    def update_params(self):
        """Copy the fused kernel into eval_conv in place (parameter identity is kept)."""
        w, b = self.fused_kernel()
        self.eval_conv.weight.copy_(w)
        if b is not None:
            self.eval_conv.bias.copy_(b)
        self._key = param_key(self)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        self.update_params()  # saved deploy kernel matches weights changed in any mode
        super()._save_to_state_dict(destination, prefix, keep_vars)

    def train(self, mode: bool = True):
        super().train(mode)
        if not mode:
            self.update_params()
        return self

    def forward(self, x):
        if fuse_online(self):
            # online re-parameterisation: the fused kernel is the same function of (x, params) as
            # 1x1 skip + 1x1-3x3-1x1 on the zero-padded input, so outputs and gradients match
            return F.conv2d(x, *self.fused_kernel(), stride=self.stride, padding=1)
        if cache_stale(self):
            self.update_params()
        return self.eval_conv(x)


class SPAB(nn.Module):
    def __init__(
        self, in_channels, mid_channels=None, out_channels=None, bias=False, fast=False
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.c1_r = Conv3XC(in_channels, mid_channels, gain1=2, s=1, bias=bias)
        self.c2_r = Conv3XC(mid_channels, mid_channels, gain1=2, s=1, bias=bias)
        self.c3_r = Conv3XC(mid_channels, out_channels, gain1=2, s=1, bias=bias)
        if not fast:
            self.act1 = torch.nn.SiLU(inplace=True)
        else:
            self.act1 = torch.nn.Mish(inplace=True)

    def forward(self, x):
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)
        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)
        out3 = self.c3_r(out2_act)
        sim_att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att
        return out, out1, sim_att


class span(nn.Module):
    """Swift Parameter-free Attention Network for Efficient Super-Resolution"""

    @staticmethod
    def _init_weights(m, mode="fan_out") -> None:
        if isinstance(m, nn.Conv2d | nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0, mode=mode, nonlinearity="linear")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        feature_channels=48,
        upscale=4,
        bias=True,
        norm=False,
        fast=False,
        img_range=1.0,
        rgb_mean=(0.5, 0.5, 0.5),
        fan_in_init=False,
    ):
        super().__init__()

        in_channels = num_in_ch
        out_channels = num_out_ch
        self.fast = fast
        self.upscale = upscale
        self.img_range = img_range
        mean = torch.tensor(rgb_mean).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)

        self.no_norm: torch.Tensor | None
        if not norm:
            self.register_buffer("no_norm", torch.zeros(1))
        else:
            self.no_norm = None

        self.block_1 = SPAB(feature_channels, bias=bias, fast=fast)
        self.block_2 = SPAB(feature_channels, bias=bias, fast=fast)
        self.block_3 = SPAB(feature_channels, bias=bias, fast=fast)
        self.block_4 = SPAB(feature_channels, bias=bias, fast=fast)
        self.block_5 = SPAB(feature_channels, bias=bias, fast=fast)

        if not self.fast:
            self.conv_1 = Conv3XC(in_channels, feature_channels, gain1=2, s=1)
            self.block_6 = SPAB(feature_channels, bias=bias, fast=fast)
        else:
            self.conv_9x9 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=feature_channels,
                kernel_size=9,
                padding=4,
                bias=bias,
            )
            self.refine_conv = nn.Conv2d(
                in_channels=feature_channels,
                out_channels=feature_channels,
                kernel_size=3,
                padding=1,
                bias=bias,
            )

        self.conv_cat = conv_layer(
            feature_channels * 4, feature_channels, kernel_size=1, bias=True
        )
        self.conv_2 = Conv3XC(
            feature_channels, feature_channels, gain1=2, s=1, bias=bias
        )

        self.upsampler = pixelshuffle_block(
            feature_channels, out_channels, upscale_factor=upscale
        )

        if self.fast and not fan_in_init:
            self.apply(self._init_weights)
        elif self.fast:
            # A-06 candidate (opt-in until an A/B): variance-preserving init with the head scaled
            # so the features start at std ~4. The bias-free SPAB gate (out3 + x) * (sigmoid(out3)
            # - 0.5) is quadratic below |out3| ~ 2: with the fan_out init the chain underflows to
            # exact zeros by block 3 (no gradient for blocks 3-5 and conv_2). conv_cat takes the
            # head features back down.
            self.apply(lambda m: self._init_weights(m, "fan_in"))
            with torch.no_grad():
                self.conv_9x9.weight.mul_(8)
                self.conv_cat.weight.mul_(1 / 8)

    @property
    def is_norm(self):
        return self.no_norm is None

    def forward(self, x):
        if self.is_norm:
            x = (x - self.mean.to(x)) * self.img_range

        if not self.fast:
            out_feature = self.conv_1(x)
        else:
            # large kernel conv
            out_feature = self.conv_9x9(x)
            # add conv to refine feature maps
            out_feature = self.refine_conv(out_feature)

        out_b1, _, _att1 = self.block_1(out_feature)
        out_b2, _, _att2 = self.block_2(out_b1)
        out_b3, _, _att3 = self.block_3(out_b2)
        out_b4, _, _att4 = self.block_4(out_b3)

        if not self.fast:
            out_b5, _, _att5 = self.block_5(out_b4)
            out_b6, out_b5_2, _att6 = self.block_6(out_b5)
            out_b6 = self.conv_2(out_b6)
            out = self.conv_cat(torch.cat([out_feature, out_b6, out_b1, out_b5_2], 1))
        else:
            out_b5, out_b4_2, _att5 = self.block_5(out_b4)
            out_b5 = self.conv_2(out_b5)
            out = self.conv_cat(torch.cat([out_feature, out_b5, out_b1, out_b4_2], 1))

        return self.upsampler(out)


def span_fast(**kwargs):
    return span(feature_channels=32, bias=False, fast=True, **kwargs)
