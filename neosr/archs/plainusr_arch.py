# type: ignore
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from torch import nn


class ASR(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.n_feat = n_feat
        self.tensor = nn.Parameter(
            0.1 * torch.ones((1, n_feat, 1, 1)), requires_grad=True
        )
        self.se = nn.Sequential(
            Reduce("b c 1 1 -> b c", "mean"),
            nn.Linear(n_feat, n_feat // 4, bias=False),
            nn.SiLU(),
            nn.Linear(n_feat // 4, n_feat, bias=False),
            nn.Sigmoid(),
            Rearrange("b c -> b c 1 1"),
        )
        self.init_weights()

    def init_weights(self):
        # to make sure the inital [0.5,0.5,...,0.5]
        self.se[1].weight.data.fill_(1)
        self.se[3].weight.data.fill_(1)

    def forward(self, x):
        attn = self.se(self.tensor)
        return attn * x


class MBConv(nn.Module):
    def __init__(self, n_feat, ratio=2):
        super().__init__()
        i_feat = n_feat * ratio
        self.expand_conv = nn.Conv2d(n_feat, i_feat, 1, 1, 0)
        self.fea_conv = nn.Conv2d(i_feat, i_feat, 3, 1, 0)
        self.reduce_conv = nn.Conv2d(i_feat, n_feat, 1, 1, 0)
        self.se = ASR(i_feat)

    def fused_kernel(self):
        """expand 1x1 -> 3x3 on the expand-bias padded map -> ASR scale + identity -> reduce 1x1
        + identity, folded into one 3x3 conv on the zero-padded input (exact, borders included:
        the bias padding is what expand gives on zero padding). fp32 under autocast."""
        with torch.autocast(self.fea_conv.weight.device.type, enabled=False):
            k0, b0 = self.expand_conv.weight[:, :, 0, 0], self.expand_conv.bias
            k1, b1 = self.fea_conv.weight, self.fea_conv.bias
            k2, b2 = self.reduce_conv.weight[:, :, 0, 0], self.reduce_conv.bias
            a = self.se.se(self.se.tensor).reshape(-1)
            w = torch.einsum("omuv,mi->oiuv", k1, k0) * a[:, None, None, None]
            b = (b1 + torch.einsum("omuv,m->o", k1, b0)) * a
            w, b = w + F.pad(k0[:, :, None, None], (1, 1, 1, 1)), b + b0
            w, b = torch.einsum("om,miuv->oiuv", k2, w), k2 @ b + b2
            eye = torch.eye(w.shape[0], dtype=w.dtype, device=w.device)
            return w + F.pad(eye[:, :, None, None], (1, 1, 1, 1)), b

    def forward(self, x):
        if "conv" in self._modules:  # deployed
            return self.conv(x)
        if torch.onnx.is_in_onnx_export():  # the TorchScript exporter needs constant kernels
            out = self.expand_conv(F.pad(x, (1, 1, 1, 1)))  # expand bias on the border
            return self.reduce_conv(self.se(self.fea_conv(out)) + out[:, :, 1:-1, 1:-1]) + x
        # online re-parameterisation: the same function of (x, params) as the branch form
        return F.conv2d(x, *self.fused_kernel(), padding=1)

    @torch.no_grad()
    def switch_to_deploy(self):
        w, b = self.fused_kernel()
        self.conv = nn.Conv2d(
            w.shape[1], w.shape[0], 3, 1, 1, device=w.device, dtype=w.dtype
        )
        self.conv.weight.copy_(w)
        self.conv.bias.copy_(b)
        del self.expand_conv, self.fea_conv, self.reduce_conv, self.se


class PConv(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.n_feat = n_feat
        self.conv = nn.Conv2d(n_feat, 1, 1, 1, 0)
        self.init_weights()

    def forward(self, x):
        x1 = self.conv(x)
        x2 = x[:, 1:].clone()
        return torch.cat([x1, x2], dim=1)

    def init_weights(self):
        self.conv.weight.data.fill_(1 / self.n_feat)
        self.conv.bias.data.fill_(0)


class SoftPooling2D(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(
            kernel_size, stride, padding, count_include_pad=False
        )

    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)
        return x / x_exp_pool


class LocalAttention(nn.Module):
    """attention based on local importance"""

    def __init__(self, channels, f=16, speed=False):
        super().__init__()
        self.speed = speed

        if self.speed:
            self.body = nn.Sequential(
                # sample importance
                nn.Conv2d(channels, channels, 3, 1, 1),
                # to heatmap
                nn.Sigmoid(),
            )
        else:
            self.body = nn.Sequential(
                # sample importance
                nn.Conv2d(channels, f, 1),
                SoftPooling2D(7, stride=3),
                nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(f, channels, 3, padding=1),
                # to heatmap
                nn.Sigmoid(),
            )

        self.gate = nn.Sequential(nn.Sigmoid())

    def forward(self, x):
        """forward"""
        # interpolate the heat map
        g = self.gate(x[:, :1].clone())
        if self.speed:
            w = self.body(x)
        else:
            w = F.interpolate(
                self.body(x),
                (x.size(2), x.size(3)),
                mode="bilinear",
                align_corners=False,
            )
        return x * w * g


class Block(nn.Module):

    def _conv_or_mb(self, n_feat):
        if self.training:
            return MBConv(n_feat)
        return nn.Conv2d(n_feat, n_feat, 3, 1, 1)

    def _conv_or_pconv(self, n_feat):
        if self.training:
            return PConv(n_feat)
        return nn.Conv2d(n_feat, n_feat, 3, 1, 1)

    def __init__(self, n_feat, version, f=16):
        super().__init__()
        self.f = f
        self.version = version

        if self.version == 1:
            self.body = nn.Sequential(
                self._conv_or_mb(n_feat),
                nn.LeakyReLU(0.05, inplace=True),
                self._conv_or_mb(n_feat),
            )
        elif self.version == 2:
            self.body = nn.Sequential(
                self._conv_or_mb(n_feat),
                nn.LeakyReLU(0.05, inplace=True),
                self._conv_or_mb(n_feat),
                self._conv_or_pconv(n_feat),
                LocalAttention(n_feat, f),
            )
        elif self.version == 3:
            self.body = nn.Sequential(
                self._conv_or_mb(n_feat),
                nn.LeakyReLU(0.05, inplace=True),
                self._conv_or_mb(n_feat),
                self._conv_or_pconv(n_feat),
                LocalAttention(n_feat, f, speed=True),
            )
        else:
            self.body = nn.Sequential(
                self._conv_or_mb(n_feat),
                nn.LeakyReLU(0.05, inplace=True),
                self._conv_or_mb(n_feat),
                nn.LeakyReLU(0.05, inplace=True),
                self._conv_or_mb(n_feat),
                self._conv_or_pconv(n_feat),
                LocalAttention(n_feat, f),
            )

    def forward(self, x):
        return self.body(x)

    @torch.no_grad()
    def switch_to_deploy(self, prune=False):
        """Plain deploy form: each MBConv becomes its fused 3x3 conv and a PConv folds into the
        conv before it (its 1x1 output replaces channel 0); activations and LocalAttention stay.
        `prune` zeroes the deployed 3x3 weights below 1e-2."""
        layers, convs = [], []
        for m in self.body:
            if isinstance(m, MBConv):
                m.switch_to_deploy()
                layers.append(m.conv)
                convs.append(m.conv)
            elif isinstance(m, PConv):
                conv, k = layers[-1], m.conv.weight[:, :, 0, 0]
                conv.weight[:1] = torch.einsum("om,miuv->oiuv", k, conv.weight)
                conv.bias[:1] = k @ conv.bias + m.conv.bias
            else:
                layers.append(m)
        if prune:
            for conv in convs:
                conv.weight.masked_fill_(conv.weight.abs() < 1e-2, 0)
        self.body = nn.Sequential(*layers)


class plainusr(nn.Module):
    """PlainUSR: Chasing Faster ConvNet for Efficient Super-Resolution:
    https://arxiv.org/abs/2409.13435
    Code adapted from:
    https://github.com/icandle/PlainUSR
    """

    def __init__(self, n_feat=64, im_feat=(64, 48, 32), attn_feat=16, scale=4):
        super().__init__()
        self.n_feat = n_feat
        self.scale = scale
        self.im_feat = im_feat
        self.head = nn.Conv2d(3, n_feat + 3, 3, 1, 1)

        if self.n_feat <= 32:
            self.block1 = Block(im_feat[0], version=1, f=attn_feat)
            self.blockm = Block(im_feat[1], version=3, f=attn_feat)
            self.block2 = Block(im_feat[0], version=1, f=attn_feat)
        elif self.n_feat >= 80:
            self.block1 = Block(im_feat[0], version=4, f=attn_feat)
            self.block2 = Block(im_feat[1], version=4, f=attn_feat)
            self.blockm = Block(im_feat[2], version=4, f=attn_feat)
            self.block3 = Block(im_feat[1], version=4, f=attn_feat)
            self.block4 = Block(im_feat[0], version=4, f=attn_feat)
        else:
            self.block1 = Block(im_feat[0], version=2, f=attn_feat)
            self.block2 = Block(im_feat[1], version=2, f=attn_feat)
            self.blockm = Block(im_feat[2], version=2, f=attn_feat)
            self.block3 = Block(im_feat[1], version=2, f=attn_feat)
            self.block4 = Block(im_feat[0], version=2, f=attn_feat)

        self.tail = nn.Sequential(
            nn.Conv2d(n_feat + 3, 3 * (scale**2), 3, 1, 1), nn.PixelShuffle(scale)
        )

        self.init_weights()

    def switch_to_deploy(self, prune=False):
        """In place: every block in its plain deploy form (same eval output unless `prune`)."""
        for m in self.children():
            if isinstance(m, Block):
                m.switch_to_deploy(prune)

    def init_weights(self):
        scale_squared = self.scale**2
        self.head.weight.data[-3, 0, 1, 1] += 1
        self.head.weight.data[-2, 1, 1, 1] += 1
        self.head.weight.data[-1, 2, 1, 1] += 1

        self.tail[0].weight.data[:scale_squared, -3, 1, 1] += 1
        self.tail[0].weight.data[scale_squared : 2 * scale_squared, -2, 1, 1] += 1
        self.tail[0].weight.data[2 * scale_squared :, -1, 1, 1] += 1

    def fast_forward(self, x):
        x = self.head(x)

        if self.n_feat <= 32:
            x[:, : self.im_feat[0]] = self.block1(x[:, : self.im_feat[0]])
            x[:, : self.im_feat[1]] = self.blockm(x[:, : self.im_feat[1]])
            x[:, : self.im_feat[0]] = self.block2(x[:, : self.im_feat[0]])
        else:
            x[:, : self.im_feat[0]] = self.block1(x[:, : self.im_feat[0]])
            x[:, : self.im_feat[1]] = self.block2(x[:, : self.im_feat[1]])
            x[:, : self.im_feat[2]] = self.blockm(x[:, : self.im_feat[2]])
            x[:, : self.im_feat[1]] = self.block3(x[:, : self.im_feat[1]])
            x[:, : self.im_feat[0]] = self.block4(x[:, : self.im_feat[0]])

        return self.tail(x)

    def forward(self, x):
        if not self.training:
            return self.fast_forward(x)

        x = self.head(x)

        if self.n_feat <= 32:
            x, pic = x.split([self.im_feat[0], 3], 1)
            x = self.block1(x)
            x1, x2 = x.split([self.im_feat[1], self.im_feat[0] - self.im_feat[1]], 1)
            x = self.block2(torch.cat([self.blockm(x1), x2], 1))  # as fast_forward
        else:
            x, pic = x.split([self.im_feat[0], 3], 1)
            x = self.block1(x)
            x1, x2 = x.split([self.im_feat[1], self.im_feat[0] - self.im_feat[1]], 1)
            x1 = self.block2(x1)
            x11, x12 = x1.split([self.im_feat[2], self.im_feat[1] - self.im_feat[2]], 1)
            x11 = self.blockm(x11)
            x1 = torch.cat([x11, x12], 1)
            x1 = self.block3(x1)
            x = torch.cat([x1, x2], 1)
            x = self.block4(x)

        x = torch.cat([x, pic], 1)
        return self.tail(x)


def plainusr_ultra(**kwargs):
    return plainusr(n_feat=32, im_feat=(32, 16), attn_feat=4, **kwargs)


def plainusr_large(**kwargs):
    return plainusr(n_feat=80, im_feat=(80, 64, 48), attn_feat=16, **kwargs)
