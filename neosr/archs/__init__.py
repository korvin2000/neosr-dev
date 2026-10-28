from collections.abc import Mapping
from typing import Any

from torch import nn

from neosr.utils.factory import build

__all__ = ["ARCHS", "build_network"]

# name -> (module under neosr.archs, ctor kwarg that takes the run scale | None for non-SR networks).
# One row per class AND factory; the scale kwarg keeps the arch's native name.
# fmt: off
ARCHS: dict[str, tuple[str, str | None]] = {
    "aether": ("aether_arch", "scale"), "aether_small": ("aether_arch", "scale"),
    "aether_medium": ("aether_arch", "scale"), "aether_large": ("aether_arch", "scale"),
    "asid": ("asid_arch", "scale"), "asid_d8": ("asid_arch", "scale"),
    "atd": ("atd_arch", "upscale"), "atd_light": ("atd_arch", "upscale"),
    "catanet": ("catanet_arch", "upscale"),
    "cfsr": ("cfsr_arch", "upscale"),
    "compact": ("compact_arch", "upscale"),
    "craft": ("craft_arch", "upscale"),
    "cugan": ("cugan_arch", "scale"),
    "dat_s": ("dat_arch", "upscale"), "dat_m": ("dat_arch", "upscale"),
    "dat_2": ("dat_arch", "upscale"),
    "dct": ("dct_arch", "upscale"),
    "dctlsa": ("dctlsa_arch", "upscale"),
    "desrgan": ("desrgan_arch", "scale"),
    "ditn": ("ditn_arch", "upscale"),
    "drct": ("drct_arch", "upscale"), "drct_l": ("drct_arch", "upscale"),
    "drct_xl": ("drct_arch", "upscale"), "drct_s": ("drct_arch", "upscale"),
    "dunet": ("dunet_arch", None),
    "ea2fpn": ("ea2fpn_arch", None),
    "eimn": ("eimn_arch", "scale"), "eimn_l": ("eimn_arch", "scale"),
    "eimn_a": ("eimn_arch", "scale"),
    "esc": ("esc_arch", "upscaling_factor"), "esc_light": ("esc_arch", "upscaling_factor"),
    "esc_fp": ("esc_arch", "upscaling_factor"), "esc_large": ("esc_arch", "upscaling_factor"),
    "esrgan": ("esrgan_arch", "scale"),
    "flexnet": ("flexnet_arch", "scale"), "metaflexnet": ("flexnet_arch", "scale"),
    "grformer": ("grformer_arch", "upscale"), "grformer_medium": ("grformer_arch", "upscale"),
    "grformer_large": ("grformer_arch", "upscale"),
    "hasn": ("hasn_arch", "upscale"),
    "hat_s": ("hat_arch", "upscale"), "hat_m": ("hat_arch", "upscale"),
    "hat_l": ("hat_arch", "upscale"),
    "hit_srf": ("hitsrf_arch", "upscale"), "hit_srf_medium": ("hitsrf_arch", "upscale"),
    "hit_srf_large": ("hitsrf_arch", "upscale"),
    "hma": ("hma_arch", "upscale"), "hma_medium": ("hma_arch", "upscale"),
    "hma_large": ("hma_arch", "upscale"),
    "krgn": ("krgn_arch", "scale"),
    "lmlt": ("lmlt_arch", "upscaling_factor"), "lmlt_tiny": ("lmlt_arch", "upscaling_factor"),
    "lmlt_large": ("lmlt_arch", "upscaling_factor"),
    "man": ("man_arch", "scale"), "man_tiny": ("man_arch", "scale"),
    "man_light": ("man_arch", "scale"),
    "metagan": ("metagan_arch", None),
    "mfghmoe": ("mfghmoe_arch", "upscale"),
    "microsr": ("microsr_arch", "upscale"), "microsr_light": ("microsr_arch", "upscale"),
    "moesr": ("moesr_arch", "scale"),
    "mosrv2": ("mosrv2_arch", "scale"),
    "msdan": ("msdan_arch", "upscale_factor"),
    "nafnet": ("nafnet_arch", "scale"), "nafnet_l": ("nafnet_arch", "scale"),  # 1x only
    "ninasr": ("ninasr_arch", "scale"), "ninasr_b0": ("ninasr_arch", "scale"),
    "ninasr_b2": ("ninasr_arch", "scale"),
    "omnisr": ("omnisr_arch", "upsampling"),
    "patchgan": ("patchgan_arch", None),
    "plainusr": ("plainusr_arch", "scale"), "plainusr_ultra": ("plainusr_arch", "scale"),
    "plainusr_large": ("plainusr_arch", "scale"),
    "plksr": ("plksr_arch", "upscaling_factor"), "plksr_tiny": ("plksr_arch", "upscaling_factor"),
    "rcan": ("rcan_arch", "scale"),
    "realplksr": ("realplksr_arch", "upscaling_factor"),
    "realplksr_s": ("realplksr_arch", "upscaling_factor"),
    "realplksr_l": ("realplksr_arch", "upscaling_factor"),
    "resgan": ("resgan_arch", None),
    "rgt": ("rgt_arch", "upscale"), "rgt_s": ("rgt_arch", "upscale"),
    "safmn": ("safmn_arch", "upscaling_factor"), "safmn_l": ("safmn_arch", "upscaling_factor"),
    "light_safmnpp": ("safmn_arch", "upscaling_factor"),
    "sebica": ("sebica_arch", "sr_rate"), "sebica_mini": ("sebica_arch", "sr_rate"),
    "span": ("span_arch", "upscale"), "span_fast": ("span_arch", "upscale"),
    "spanplus": ("spanplus_arch", "upscale"), "spanplus_sts": ("spanplus_arch", "upscale"),
    "spanplus_s": ("spanplus_arch", "upscale"), "spanplus_st": ("spanplus_arch", "upscale"),
    "srformer": ("srformer_arch", "upscale"), "srformer_light": ("srformer_arch", "upscale"),
    "srformer_medium": ("srformer_arch", "upscale"),
    "swinir_small": ("swinir_arch", "upscale"), "swinir_medium": ("swinir_arch", "upscale"),
    "swinir_large": ("swinir_arch", "upscale"),
    "swinirdense": ("swinirdense_arch", "upscale"),
    "unet": ("unet_arch", None),
    "VGGFeatureExtractor": ("vgg_arch", None),
}
# fmt: on


def build_network(
    cfg: Mapping[str, Any], *, scale: int | None = None, section: str = "network_g"
) -> nn.Module:
    return build(ARCHS, "neosr.archs", section, cfg, scale=scale)
