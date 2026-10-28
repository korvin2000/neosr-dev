"""NAFNet (Chen et al., "Simple Baselines for Image Restoration", ECCV 2022), a 1x restoration U-Net
written from the paper and the official MIT code (megvii-research/NAFNet) with its key names, so the
official `.pth` files load. LayerNorm2d is F.layer_norm on the NHWC view (arch_util), not the
official custom autograd Function (same math; fp32 under autocast, compile-friendly)."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from neosr.archs.arch_util import LayerNorm2d
from neosr.config import ConfigError


class SimpleGate(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(
        self,
        c: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        drop_out_rate: float = 0.0,
    ) -> None:
        super().__init__()
        dw = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.conv3 = nn.Conv2d(dw // 2, c, 1)
        # simplified channel attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1)
        )
        self.sg = SimpleGate()
        ffn = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)
        self.norm1, self.norm2 = LayerNorm2d(c, eps=1e-6), LayerNorm2d(c, eps=1e-6)
        self.dropout1 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        )
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp: Tensor) -> Tensor:
        x = self.sg(self.conv2(self.conv1(self.norm1(inp))))
        x = self.dropout1(self.conv3(x * self.sca(x)))
        y = inp + x * self.beta
        x = self.conv5(self.sg(self.conv4(self.norm2(y))))
        return y + self.dropout2(x) * self.gamma


class nafnet(nn.Module):
    """Defaults: the official SIDD width-32 model (enc [2, 2, 4, 8], middle 12, dec [2, 2, 2, 2])."""

    def __init__(
        self,
        img_channel: int = 3,
        width: int = 32,
        middle_blk_num: int = 12,
        enc_blk_nums: tuple[int, ...] = (2, 2, 4, 8),
        dec_blk_nums: tuple[int, ...] = (2, 2, 2, 2),
        drop_out_rate: float = 0.0,
        scale: int = 1,
    ) -> None:
        super().__init__()
        if scale != 1:
            msg = f"nafnet is a 1x restoration network: set scale = 1 (got {scale})"
            raise ConfigError(msg)
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1)
        self.encoders, self.downs = nn.ModuleList(), nn.ModuleList()
        self.decoders, self.ups = nn.ModuleList(), nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[
                    NAFBlock(chan, drop_out_rate=drop_out_rate) for _ in range(num)
                ])
            )
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2
        self.middle_blks = nn.Sequential(*[
            NAFBlock(chan, drop_out_rate=drop_out_rate) for _ in range(middle_blk_num)
        ])
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)
                )
            )
            chan //= 2
            self.decoders.append(
                nn.Sequential(*[
                    NAFBlock(chan, drop_out_rate=drop_out_rate) for _ in range(num)
                ])
            )
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp: Tensor) -> Tensor:
        h, w = inp.shape[-2:]
        p = self.padder_size
        x = F.pad(inp, (0, (p - w % p) % p, 0, (p - h % p) % p))
        skip = x
        x = self.intro(x)
        encs = []
        for encoder, down in zip(self.encoders, self.downs, strict=True):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, enc in zip(self.decoders, self.ups, encs[::-1], strict=True):
            x = decoder(up(x) + enc)
        return (self.ending(x) + skip)[..., :h, :w]


def nafnet_l(**kwargs) -> nafnet:
    """The official width-64 model (same block counts)."""
    return nafnet(width=64, **kwargs)
