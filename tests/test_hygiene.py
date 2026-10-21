"""Banned constructs under neosr/. ALLOW lists the files not migrated yet; lanes shrink it as they land."""

import re
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "neosr"

PATTERNS = {
    "net_opt": r"\bnet_opt\b",
    "parse_options": r"\bparse_options\b",
    "set_default_device": r"\bset_default_device\b",
    "sys.exit": r"\bsys\.exit\b",
    "sys.argv": r"\bsys\.argv\b",
    "bare except": r"^\s*except\s*:",
    "registry": r"_REGISTRY\b",
}

_SR = (
    "asid atd catanet cfsr compact craft cugan dat dct dctlsa desrgan ditn drct eimn esc esrgan flexnet "
    "grformer hasn hat hitsrf hma krgn lmlt man mfghmoe microsr moesr mosrv2 msdan ninasr omnisr plainusr "
    "plksr rcan realplksr rgt safmn sebica span spanplus srformer swinir swinirdense"
).split()
_NON_SR = "dunet ea2fpn metagan patchgan resgan unet vgg".split()
_LOSSES = (
    "basic consistency dists fdl ff gan kl ldl msswd ncc ssim vgg_perceptual"
).split()

# pattern -> files (relative to neosr/) that may still contain it
ALLOW: dict[str, set[str]] = {
    "net_opt": {"archs/arch_util.py", *(f"archs/{a}_arch.py" for a in _SR)},
    "parse_options": {
        "archs/arch_util.py",
        "optimizers/soap_sf.py",
        "utils/options.py",
        "utils/rng.py",
    },
    "set_default_device": set(),
    "sys.exit": {
        "models/base.py",
        "models/image.py",
        "utils/lmdb_util.py",
        "utils/options.py",
    },
    "sys.argv": {"utils/options.py"},
    "bare except": {
        "archs/arch_util.py",
        "archs/esrgan_arch.py",
        "models/base.py",
        "utils/img_util.py",
        "utils/options.py",
    },
    "registry": {
        *(f"archs/{a}_arch.py" for a in _SR + _NON_SR),
        "archs/__init__.py",
        "data/__init__.py",
        "data/otf_dataset.py",
        "data/paired_dataset.py",
        "data/single_dataset.py",
        *(f"losses/{loss}_loss.py" for loss in _LOSSES),
        "losses/__init__.py",
        "metrics/__init__.py",
        "metrics/calculate.py",
        "metrics/topiq.py",
        "models/__init__.py",
        "models/image.py",
        "models/otf.py",
        "utils/registry.py",
    },
}


def offenders(pattern: str) -> set[str]:
    regex = re.compile(PATTERNS[pattern], re.MULTILINE)
    return {
        str(f.relative_to(PKG))
        for f in PKG.rglob("*.py")
        if regex.search(f.read_text("utf-8"))
    }


@pytest.mark.parametrize("pattern", PATTERNS)
def test_no_new_offenders(pattern: str) -> None:
    assert sorted(offenders(pattern) - ALLOW[pattern]) == []


@pytest.mark.parametrize("pattern", PATTERNS)
def test_allowlist_is_current(pattern: str) -> None:
    """A migrated or deleted file must leave the allowlist, so the list only shrinks."""
    assert sorted(ALLOW[pattern] - offenders(pattern)) == []
