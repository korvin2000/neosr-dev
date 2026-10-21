"""
AetherNet: High-Performance Super-Resolution Architecture
Core implementation optimized for quality, speed, and deployment

Key Improvements:
1. Fixed channel dimension handling throughout
2. Corrected statistical fusion in DeploymentNorm
3. Optimized quantization workflow
4. Enhanced efficiency with convolutional attention
5. Scale-aware upsampling with proper channel management
6. TensorRT/ONNX export optimizations

Author: Philip Hofmann
License: MIT
GitHub: https://github.com/phhofm/aethernet

neosr keeps this (v4) line as the only implementation (docs/aether.md). QAT contract (design T5), used
by neosr/models/image.py and convert.py: `prepare_qat()` once on the bare float net, `set_qat_phase()`
per update, `qat_state(sd)` rebuilds the topology of a QAT state dict, `fold_norms()` / `fuse_model()`
give the deploy graph.
"""

import math
from collections.abc import Mapping
from typing import Any, List, Tuple

import torch
import torch.ao.quantization as tq
from torch import nn
from torch.ao.nn.quantized import FloatFunctional
from torch.nn import functional as F
from torch.nn.init import trunc_normal_

SCALES = (2, 3, 4, 8)
# image-domain convs (and the input stub) stay float under QAT (T5 D8)
FLOAT_LAYERS = ("quant", "conv_first", "conv_before_upsample.0", "conv_last")


class NarrowRangeObserver(tq.MovingAveragePerChannelMinMaxObserver):
    """Per-channel scale max|w| / 127 instead of / 127.5: the extreme weight of a channel lands on
    +-127, not on the +-127.5 rounding tie that torch and ONNX Runtime break differently. The ONNX
    exporter accepts only the (-128, 127) range, so the range stays; code -128 goes unused."""

    def calculate_qparams(self):  # type: ignore[override]
        scale, zero_point = super().calculate_qparams()
        return scale * (127.5 / 127), zero_point


def qat_qconfig(qat_type: str = "int8") -> tq.QConfig:
    """Symmetric int8 for TensorRT / ONNX Runtime QDQ: per-tensor activations, per-channel weights
    (T5 D7). Plain FakeQuantize (not the fused one): it has ONNX symbolics and runs under autocast."""
    if qat_type != "int8":
        msg = f"qat_type={qat_type!r} is not supported; use 'int8'"
        raise ValueError(msg)
    act = tq.FakeQuantize.with_args(
        observer=tq.MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
    )
    weight = tq.FakeQuantize.with_args(
        observer=NarrowRangeObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
    )
    return tq.QConfig(activation=act, weight=weight)


# ------------------- Core Building Blocks ------------------- #

class DropPath(nn.Module):
    """Stochastic Depth with ONNX-compatible implementation"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class ReparamLargeKernelConv(nn.Module):
    """Efficient large kernel convolution with TRT optimization"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, groups: int, small_kernel: int, fused_init: bool = False):
        super().__init__()
        if kernel_size % 2 == 0 or small_kernel % 2 == 0:
            raise ValueError("Kernel sizes must be odd numbers")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = kernel_size // 2
        self.small_kernel = small_kernel
        self.fused = fused_init
        self.is_quantized = False

        # conv padding, not a ZeroPad2d: no dynamic Pad between Q/DQ and the conv in ONNX
        if self.fused:
            self.fused_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride,
                self.padding, groups=groups, bias=True
            )
        else:
            self.lk_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride,
                self.padding, groups=groups, bias=False
            )
            self.sk_conv = nn.Conv2d(
                in_channels, out_channels, small_kernel, stride,
                small_kernel//2, groups=groups, bias=False
            )
            self.lk_bias = nn.Parameter(torch.zeros(out_channels))
            self.sk_bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            return self.fused_conv(x)

        lk_out = self.lk_conv(x)
        sk_out = self.sk_conv(x)
        return (lk_out + self.lk_bias.view(1, -1, 1, 1) +
                sk_out + self.sk_bias.view(1, -1, 1, 1))

    def _fuse_kernel(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse large and small kernels into single convolution"""
        if self.fused:
            raise RuntimeError("Already fused")

        pad = (self.kernel_size - self.small_kernel) // 2
        sk_kernel_padded = F.pad(self.sk_conv.weight, [pad]*4)

        fused_kernel = self.lk_conv.weight + sk_kernel_padded
        fused_bias = self.lk_bias + self.sk_bias
        return fused_kernel, fused_bias

    def fuse(self):
        """Fuse components into single convolution for deployment"""
        if self.fused:
            return

        fused_kernel, fused_bias = self._fuse_kernel()
        self.fused_conv = nn.Conv2d(
            self.in_channels, self.out_channels, self.kernel_size,
            self.stride, self.padding, groups=self.groups, bias=True
        )
        self.fused_conv.weight.data = fused_kernel
        self.fused_conv.bias.data = fused_bias

        # Propagate quantization settings
        if self.is_quantized:
            self.fused_conv.qconfig = self.lk_conv.qconfig

        # Clean up unused parameters
        del self.lk_conv, self.sk_conv, self.lk_bias, self.sk_bias
        self.fused = True

class GatedConvFFN(nn.Module):
    """Convolution-based Gated FFN with temperature scaling"""
    def __init__(self, in_channels: int, mlp_ratio: float = 2.0, drop: float = 0.):
        super().__init__()
        hidden_channels = int(in_channels * mlp_ratio)

        self.conv_gate = nn.Conv2d(in_channels, hidden_channels, 1)
        self.conv_main = nn.Conv2d(in_channels, hidden_channels, 1)
        self.act = nn.SiLU()
        self.quant_act = tq.QuantStub()  # identity until prepare_qat (T5 D6)
        self.drop1 = nn.Dropout(drop)
        self.conv_out = nn.Conv2d(hidden_channels, in_channels, 1)
        self.drop2 = nn.Dropout(drop)

        self.quant_mul = FloatFunctional()
        self.is_quantized = False
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.conv_gate(x) / self.temperature
        main = self.conv_main(x)
        activated = self.quant_act(self.act(gate))

        if self.is_quantized:
            x = self.quant_mul.mul(activated, main)
        else:
            x = activated * main

        x = self.drop1(x)
        x = self.conv_out(x)
        return self.drop2(x)

class DynamicChannelScaling(nn.Module):
    """Efficient channel attention using 1x1 convolutions"""
    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid()
        )
        self.quant_mul = FloatFunctional()
        self.is_quantized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        scale = self.fc(y)

        if self.is_quantized:
            return self.quant_mul.mul(x, scale)
        return x * scale

class SpatialAttention(nn.Module):
    """Lightweight spatial attention with TRT optimizations"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.quant_mul = FloatFunctional()
        self.is_quantized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        concat = torch.cat([max_pool, avg_pool], dim=1)
        attention_map = self.sigmoid(self.conv(concat))

        if self.is_quantized:
            return self.quant_mul.mul(x, attention_map)
        return x * attention_map

class DeploymentNorm(nn.Module):
    """ONNX-friendly normalization with proper statistical fusion. `fused_flag` persists the folded
    state (T5 D19), so a file loads into the behaviour it was saved with; files without it keep the
    constructor's `fused`."""
    def __init__(self, channels: int, fused: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = 1e-5
        self.fused = fused

        # Buffers for running statistics
        self.register_buffer('running_mean', torch.zeros(1, channels, 1, 1))
        self.register_buffer('running_var', torch.ones(1, channels, 1, 1))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.register_buffer('fused_flag', torch.tensor(fused))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        state_dict.setdefault(prefix + "fused_flag", self.fused_flag.clone())
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self.fused = bool(self.fused_flag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            return x * self.weight + self.bias

        if self.training:
            # Calculate per-channel statistics
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)

            # Update running stats with exponential moving average
            with torch.no_grad():
                self.running_mean.mul_(0.9).add_(mean, alpha=0.1)
                self.running_var.mul_(0.9).add_(var, alpha=0.1)
                self.num_batches_tracked += 1
        else:
            mean = self.running_mean
            var = self.running_var

        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias

    def fuse(self):
        """Fuse normalization into affine parameters for deployment"""
        if self.fused:
            return

        # Compute fused scale and shift
        scale = self.weight / torch.sqrt(self.running_var + self.eps)
        shift = self.bias - self.running_mean * scale

        # Update parameters
        self.weight.data = scale
        self.bias.data = shift
        self.fused = True
        self.fused_flag.fill_(True)


class AetherBlock(nn.Module):
    """Core building block with enhanced quantization control"""
    def __init__(self, dim: int, mlp_ratio: float = 2.0, drop: float = 0.,
                 drop_path: float = 0., lk_kernel: int = 11, sk_kernel: int = 3,
                 fused_init: bool = False, quantize_residual: bool = True,
                 use_channel_attn: bool = True, use_spatial_attn: bool = True):
        super().__init__()
        self.conv = ReparamLargeKernelConv(
            in_channels=dim, out_channels=dim, kernel_size=lk_kernel,
            stride=1, groups=dim, small_kernel=sk_kernel, fused_init=fused_init
        )
        self.norm = DeploymentNorm(dim, fused=fused_init)
        self.quant_norm = tq.QuantStub()
        self.ffn = GatedConvFFN(in_channels=dim, mlp_ratio=mlp_ratio, drop=drop)
        self.channel_attn = DynamicChannelScaling(dim) if use_channel_attn else nn.Identity()
        self.spatial_attn = SpatialAttention() if use_spatial_attn else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # prepare_qat() may override it; the stubs hold no state, so the float keys never change
        self.quantize_residual = quantize_residual
        self.quant_add = FloatFunctional()
        self.is_quantized = False
        self.residual_quant = tq.QuantStub()
        self.residual_dequant = tq.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv(x)
        x = self.quant_norm(self.norm(x))
        x = self.ffn(x)
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        residual = self.drop_path(x)

        # Handle residual connection with quantization
        if self.is_quantized and self.quantize_residual:
            shortcut_quant = self.residual_quant(shortcut)
            output = self.quant_add.add(shortcut_quant, residual)
            return self.residual_dequant(output)

        return shortcut + residual

class QuantFusion(nn.Module):
    """Fusion layer with consistent channel dimension handling"""
    def __init__(self, channels: int, num_inputs: int):
        super().__init__()
        # Maintain original channel dimension
        self.fusion_conv = nn.Conv2d(channels, channels, 1)

        # Error compensation tensor
        self.error_comp = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # Quantization operations
        self.quant_add = FloatFunctional()
        self.is_quantized = False

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # Align feature resolutions
        target_size = features[0].shape[-2:]
        aligned_features = []
        for feat in features:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size,
                                     mode='bilinear', align_corners=False)
            aligned_features.append(feat)

        # Concatenate along channel dimension
        x = torch.cat(aligned_features, dim=1)

        # Apply quantization-aware error compensation
        if self.is_quantized:
            x = self.quant_add.add(x, self.error_comp)
        else:
            x = x + self.error_comp

        return self.fusion_conv(x)


class AdaptiveUpsample(nn.Module):
    """Resolution-aware upsampling with automatic channel management"""
    def __init__(self, scale: int, in_channels: int):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.blocks = nn.ModuleList()

        # Calculate output channels based on scale
        self.out_channels = max(32, in_channels // max(1, scale // 2))

        # Power of 2 scaling
        if (scale & (scale - 1)) == 0:
            num_ups = int(math.log2(scale))
            # Initial upscaling
            self.blocks.append(nn.Conv2d(
                in_channels, 4 * self.out_channels, 3, 1, 1
            ))
            self.blocks.append(nn.PixelShuffle(2))

            # Additional upscaling steps
            for _ in range(num_ups - 1):
                self.blocks.append(nn.Conv2d(
                    self.out_channels, 4 * self.out_channels, 3, 1, 1
                ))
                self.blocks.append(nn.PixelShuffle(2))
        elif scale == 3:
            # 3x scaling
            self.blocks.append(nn.Conv2d(
                in_channels, 9 * self.out_channels, 3, 1, 1
            ))
            self.blocks.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"Unsupported scale: {scale}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

# ------------------- Main AetherNet Architecture ------------------- #

class aether(nn.Module):
    """AetherNet: Production-Ready Super-Resolution Network. `fused_init=True` builds the deploy
    topology (fused reparam convs, folded norms) for loading a fuse_model() checkpoint."""
    def _init_weights(self, m: nn.Module):
        """Weight initialization for different layer types"""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.GroupNorm, DeploymentNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (4, 4, 4, 4),
        mlp_ratio: float = 2.0,
        drop: float = 0.0,
        drop_path_rate: float = 0.1,
        lk_kernel: int = 11,
        sk_kernel: int = 3,
        scale: int = 4,
        img_range: float = 1.0,
        fused_init: bool = False,
        quantize_residual: bool = True,
        use_channel_attn: bool = True,
        use_spatial_attn: bool = True,
    ):
        super().__init__()
        if scale not in SCALES:  # AdaptiveUpsample would build scale 1 as 2x
            msg = f"aether: scale={scale} is not supported; use one of {SCALES}"
            raise ValueError(msg)
        if embed_dim % len(depths):
            msg = f"aether: embed_dim={embed_dim} must be divisible by len(depths)={len(depths)}"
            raise ValueError(msg)
        self.img_range = img_range
        self.scale = scale
        self.fused_init = fused_init
        self.embed_dim = embed_dim
        self.quantize_residual = quantize_residual
        self.use_channel_attn = use_channel_attn
        self.use_spatial_attn = use_spatial_attn
        self.num_stages = len(depths)
        self.is_quantized = False  # Global quantization flag

        # Input normalization parameters
        self.register_buffer('mean', torch.full((1, in_chans, 1, 1), 0.5))

        # --- Initial Feature Extraction ---
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.quant_first = tq.QuantStub()

        # --- Deep Feature Processing ---
        self.stages = nn.ModuleList()
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # Multi-scale feature fusion
        self.fusion_convs = nn.ModuleList()

        block_idx = 0
        for i, depth in enumerate(depths):
            stage_blocks = []
            for j in range(depth):
                block = AetherBlock(
                    dim=embed_dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=dpr[block_idx + j],
                    lk_kernel=lk_kernel,
                    sk_kernel=sk_kernel,
                    fused_init=fused_init,
                    quantize_residual=self.quantize_residual,
                    use_channel_attn=self.use_channel_attn,
                    use_spatial_attn=self.use_spatial_attn
                )
                stage_blocks.append(block)
            self.stages.append(nn.Sequential(*stage_blocks))
            block_idx += depth

            # Reduce channel dimension for fusion
            self.fusion_convs.append(nn.Conv2d(embed_dim, embed_dim // self.num_stages, 1))

        # Quantization-safe feature fusion
        self.quant_fusion_layer = QuantFusion(
            channels=embed_dim,  # Maintain original dimension
            num_inputs=self.num_stages
        )

        # --- Post-Processing ---
        self.norm = DeploymentNorm(embed_dim, fused=fused_init)
        self.quant_body = tq.QuantStub()
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # --- Reconstruction ---
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )

        # Initialize upsampler with correct input channels
        self.upsample = AdaptiveUpsample(scale, in_channels=embed_dim // 2)
        self.conv_last = nn.Conv2d(self.upsample.out_channels, in_chans, 3, 1, 1)

        # fused models too: a fresh deploy topology must not hold uninitialised memory
        self.apply(self._init_weights)

        # Quantization stubs
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input normalization
        x = (x - self.mean) * self.img_range

        # Quantize input if in quantized mode
        if self.is_quantized:
            x = self.quant(x)

        # --- Shallow Features ---
        x = self.quant_first(self.conv_first(x))
        shortcut = x

        # --- Deep Features ---
        features = []
        for stage, fusion_conv in zip(self.stages, self.fusion_convs):
            x = stage(x)
            features.append(fusion_conv(x))

        # Fuse multi-scale features
        x = self.quant_fusion_layer(features)
        x = x + shortcut

        # --- Post-Processing ---
        x = self.conv_after_body(self.quant_body(self.norm(x))) + x

        # --- Reconstruction ---
        x = self.conv_before_upsample(x)
        x = self.upsample(x)
        x = self.conv_last(x)

        # Dequantize if needed
        if self.is_quantized:
            x = self.dequant(x)

        # Output denormalization
        return x / self.img_range + self.mean

    # ------------------- deployment and QAT (design T5) ------------------- #
    def fuse_reparam(self) -> None:
        for m in [*self.modules()]:
            if isinstance(m, ReparamLargeKernelConv):
                m.fuse()

    def fold_norms(self) -> None:
        for m in self.modules():
            if isinstance(m, DeploymentNorm):
                m.fuse()

    def fuse_model(self) -> None:
        """Deploy topology, in place and idempotent: reparam convs fused, norms folded."""
        self.fuse_reparam()
        self.fold_norms()
        self.fused_init = True

    def prepare_qat(self, qat_type: str = "int8", quantize_residual: bool | None = None) -> None:
        """Eager-mode QAT topology, in place on the module's device (T5 D5-D10). Reparam convs are
        fused first; DeploymentNorms stay live (folded only at export), so QAT works from scratch.
        Leaves the net idle (fake quant and observers off): the caller picks the phase."""
        if self.is_quantized:
            msg = "prepare_qat() was already called"
            raise RuntimeError(msg)
        if quantize_residual is not None:
            self.quantize_residual = quantize_residual
        self.fuse_reparam()
        self.qconfig = qat_qconfig(qat_type)
        for name in FLOAT_LAYERS:
            self.get_submodule(name).qconfig = None
        for m in self.modules():  # torch would give Sigmoid a fixed-qparams uint8 Q/DQ (no TensorRT)
            if isinstance(m, nn.Sigmoid):
                m.qconfig = None
        for blk in self.modules():
            if isinstance(blk, AetherBlock):
                blk.quantize_residual = self.quantize_residual
                if not self.quantize_residual:  # the residual add stays float
                    for m in (blk.residual_quant, blk.residual_dequant, blk.quant_add):
                        m.qconfig = None
        was = self.training
        tq.prepare_qat(self.train(), inplace=True)
        for m in self.modules():
            if hasattr(m, "is_quantized"):
                m.is_quantized = True
        # materialise the per-channel scale shapes (EMA and DDP copy buffers in place) without
        # moving the norm statistics (eval), then forget the dummy activation ranges (T5 D10)
        self.eval()
        with torch.no_grad():
            self(torch.rand(1, self.mean.shape[1], 16, 16, device=self.mean.device))
        for m in self.modules():
            obs = getattr(m, "activation_post_process", None)
            if isinstance(m, tq.FakeQuantize) and isinstance(obs, tq.MinMaxObserver):
                obs.reset_min_max_vals()
        self.train(was)
        self.set_qat_phase(fake_quant=False, observe=False, freeze_norm=False)

    def set_qat_phase(self, fake_quant: bool, observe: bool, freeze_norm: bool) -> None:
        """Fake quant on/off, observers on/off, DeploymentNorm statistics frozen (eval) or live."""
        self.apply(tq.enable_fake_quant if fake_quant else tq.disable_fake_quant)
        self.apply(tq.enable_observer if observe else tq.disable_observer)
        for m in self.modules():
            if isinstance(m, DeploymentNorm):
                m.train(self.training and not freeze_norm)

    @staticmethod
    def qat_state(sd: Mapping[str, Any]) -> dict[str, Any] | None:
        """prepare_qat() kwargs that rebuild the topology of a QAT state dict; None for a float one."""
        if not any("activation_post_process." in k or "weight_fake_quant." in k for k in sd):
            return None
        key = ".residual_quant.activation_post_process."
        return {"quantize_residual": any(key in k for k in sd)}


# ------------------- Recommended Configurations ------------------- #
def aether_small(**kwargs) -> aether:
    return aether(embed_dim=96, depths=(4, 4, 4, 4), use_channel_attn=False,
                  use_spatial_attn=False, **kwargs)


def aether_medium(**kwargs) -> aether:
    return aether(embed_dim=128, depths=(6, 6, 6, 6), use_channel_attn=True,
                  use_spatial_attn=True, **kwargs)


def aether_large(**kwargs) -> aether:
    return aether(embed_dim=180, depths=(8, 8, 8, 8, 8), use_channel_attn=True,
                  use_spatial_attn=True, **kwargs)
