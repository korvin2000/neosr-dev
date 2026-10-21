# type: ignore
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import trunc_normal_

from neosr.archs.arch_util import DySample
from neosr.archs.span_arch import fuse_conv3xc


class Conv3XC(nn.Module):
    def __init__(
        self, c_in: int, c_out: int, gain: int = 1, s: int = 1, bias: bool = True
    ):
        super(Conv3XC, self).__init__()
        self.stride = s

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
        self.eval_conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=3,
            padding=1,
            stride=s,
            bias=bias,
        )
        trunc_normal_(self.sk.weight, std=0.02)
        trunc_normal_(self.eval_conv.weight, std=0.02)
        # deploy conv (export): fused kernel, refreshed on eval() and state_dict()
        self.eval_conv.requires_grad_(requires_grad=False)

    def fused_kernel(self):
        return fuse_conv3xc(self.sk, self.conv)

    @torch.no_grad()
    def update_params(self):
        """Copy the fused kernel into eval_conv in place (parameter identity is kept)."""
        w, b = self.fused_kernel()
        self.eval_conv.weight.copy_(w)
        if b is not None:
            self.eval_conv.bias.copy_(b)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        self.update_params()  # saved deploy kernel matches weights changed in any mode
        super()._save_to_state_dict(destination, prefix, keep_vars)

    def train(self, mode: bool = True):
        super().train(mode)
        if not mode:
            self.update_params()
        return self

    def forward(self, x):
        if self.training:
            return self.conv(F.pad(x, (1, 1, 1, 1), "constant", 0)) + self.sk(x)
        # fused on the fly: never stale, whatever changed the branch weights in eval mode
        return F.conv2d(x, *self.fused_kernel(), stride=self.stride, padding=1)


class SPAB(nn.Module):
    def __init__(self, in_channels: int, end: bool = False):
        super(SPAB, self).__init__()

        self.in_channels = in_channels
        self.c1_r = Conv3XC(in_channels, in_channels, gain=2, s=1)
        self.c2_r = Conv3XC(in_channels, in_channels, gain=2, s=1)
        self.c3_r = Conv3XC(in_channels, in_channels, gain=2, s=1)
        self.act1 = nn.Mish(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.end = end

    def forward(self, x):
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)

        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)

        out3 = self.c3_r(out2_act)

        sim_att = self.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att
        if self.end:
            return out, out1
        return out


class SPABS(nn.Module):
    def __init__(self, feature_channels: int, n_blocks: int = 4, drop: float = 0.0):
        super(SPABS, self).__init__()
        self.block_1 = SPAB(feature_channels)

        self.block_n = nn.Sequential(*[SPAB(feature_channels) for _ in range(n_blocks)])
        self.block_end = SPAB(feature_channels, True)
        self.conv_2 = Conv3XC(feature_channels, feature_channels, gain=2, s=1)
        self.conv_cat = nn.Conv2d(
            feature_channels * 4, feature_channels, kernel_size=1, bias=True
        )
        self.dropout = nn.Dropout2d(drop)
        trunc_normal_(self.conv_cat.weight, std=0.02)

    def forward(self, x):
        out_b1 = self.block_1(x)
        out_x = self.block_n(out_b1)
        out_end, out_x_2 = self.block_end(out_x)
        out_end = self.dropout(self.conv_2(out_end))
        return self.conv_cat(torch.cat([x, out_end, out_b1, out_x_2], 1))


class spanplus(nn.Module):
    """Modified from 'Swift Parameter-free Attention Network for Efficient Super-Resolution':
    https://arxiv.org/abs/2311.12770
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        blocks: list = [4],
        feature_channels: int = 48,
        upscale: int = 4,
        drop_rate: float = 0.0,
        upsampler: str = "dys",  # "lp", "ps", "conv"- only 1x
    ):
        super(spanplus, self).__init__()

        in_channels = num_in_ch
        out_channels = num_out_ch
        if not isinstance(blocks, list):
            blocks = [int(blocks)]
        self.feats = nn.Sequential(
            *[Conv3XC(in_channels, feature_channels, gain=2, s=1)]
            + [SPABS(feature_channels, n_blocks, drop_rate) for n_blocks in blocks]
        )
        if upsampler == "ps":
            self.upsampler = nn.Sequential(
                nn.Conv2d(feature_channels, out_channels * (upscale**2), 3, padding=1),
                nn.PixelShuffle(upscale),
            )
        elif upsampler == "dys":
            self.upsampler = DySample(feature_channels, out_channels, upscale)
        elif upsampler == "conv":
            if upscale != 1:
                msg = "conv supports only 1x"
                raise ValueError(msg)

            self.upsampler = nn.Conv2d(feature_channels, out_channels, 3, padding=1)
        else:
            raise NotImplementedError(
                f'upsampler: {upsampler} not supported, choose one of these options: \
                ["ps", "dys", "conv"] conv supports only 1x'
            )

    def forward(self, x):
        out = self.feats(x)
        return self.upsampler(out)


def spanplus_sts(**kwargs):
    return spanplus(blocks=[2], feature_channels=32, upsampler="ps", **kwargs)


def spanplus_s(**kwargs):
    return spanplus(blocks=[2], feature_channels=32, **kwargs)


def spanplus_st(**kwargs):
    return spanplus(upsampler="ps", **kwargs)
