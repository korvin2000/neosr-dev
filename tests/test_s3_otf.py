"""Stage 3 OTF realism (docs/refactor/stage3/otf/DESIGN.md). O1: the antialiased resampler and the
`resize_mode*` keys of the Degrader. O2: per-sample noise/JPEG gates, the clean path and the
`blur_prob` of the dataset worker. CPU, tiny tensors."""

import math
import random
import re
from fractions import Fraction
from itertools import starmap

import pytest
import torch
from torch.nn import functional as F

from neosr import config
from neosr.config import ConfigError
from neosr.data import build_dataset
from neosr.data import otf_degradation as od
from neosr.data.otf_degradation import Degrader
from neosr.data.resample import KERNELS, resample, scaled_size
from neosr.utils import rng as rng_util
from neosr.utils.diffjpeg import DiffJPEG, filter2D
from tests import test_otf_degradation as base
from tests import test_s2_data as data

pairs = data.pairs  # the fixture: three x4 GT/LQ pairs in tmp_path

CFG, S, _batch, _seed = base.CFG, base.S, base._batch, base._seed

SIZES = [(9, 16), (18, 31), (37, 64), (50, 97), (12, 100), (1, 1), (37, 5)]
MODE_KEYS = {
    "resize_mode": "resize_mode_prob",
    "resize_mode2": "resize_mode_prob2",
    "final_resize_mode": "final_resize_mode_prob",
}


def _x(dtype=torch.float64, h=37, w=64):
    g = torch.Generator().manual_seed(0)
    return torch.rand(2, 3, h, w, generator=g, dtype=dtype)


# ---------------------------------------------------------------- resampler
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize(
    ("kernel", "mode"), [("aa_bicubic", "bicubic"), ("aa_bilinear", "bilinear")]
)
def test_equals_torch_antialias(size, kernel, mode):
    exact = F.interpolate(_x(), size=size, mode=mode, antialias=True)
    # torch builds its fp32 weights in fp32 (4e-6 off its own fp64 result on an upscale); the
    # weights here are fp64 cast once, so the fp32 result is held against the fp64 reference too
    for dtype, tol, tol_exact in (
        (torch.float64, 1e-12, 1e-12),
        (torch.float32, 1e-5, 2e-6),
    ):
        x = _x().to(dtype)
        ref = F.interpolate(x, size=size, mode=mode, antialias=True)
        got = resample(x, size, kernel)
        assert got.dtype == dtype
        assert got.shape == ref.shape
        assert (got - ref).abs().max() <= tol
        assert (got.double() - exact).abs().max() <= tol_exact


def _scalar(name, x):
    """The kernels spelled out for one offset (an exact Fraction), independent of the tensor
    code; the box decides its half-open boundary exactly."""
    if name == "aa_box":
        return float(Fraction(-1, 2) <= x < Fraction(1, 2))
    x = float(x)
    ax = abs(x)
    if name == "aa_bilinear":
        return max(0.0, 1 - ax)
    if name == "aa_bicubic":
        a = -0.5
        if ax <= 1:
            return (a + 2) * ax**3 - (a + 3) * ax**2 + 1
        return a * ax**3 - 5 * a * ax**2 + 8 * a * ax - 4 * a if ax < 2 else 0.0
    if name == "aa_mitchell":
        if ax < 1:
            return (7 * ax**3 - 12 * ax**2 + 16 / 3) / 6
        return (-7 / 3 * ax**3 + 12 * ax**2 - 20 * ax + 32 / 3) / 6 if ax < 2 else 0.0
    a = int(name[-1])  # lanczos
    if ax >= a:
        return 0.0
    if x == 0:
        return 1.0
    return a * math.sin(math.pi * x) * math.sin(math.pi * x / a) / (math.pi * x) ** 2


def _reference_1d(row, m, name):
    """The PIL rule for one row: stretch when downscaling, truncate at the border, renormalise."""
    n = len(row)
    scale = Fraction(n, m)
    out = []
    for i in range(m):
        centre = Fraction(2 * i + 1, 2) * scale - Fraction(1, 2)
        ws = [_scalar(name, (j - centre) / max(scale, 1)) for j in range(n)]
        out.append(sum(w * v for w, v in zip(ws, row, strict=True)) / sum(ws))
    return out


@pytest.mark.parametrize("name", sorted(KERNELS))
# not 23: an unchanged axis is returned as is; 22 and 46 put samples on box boundaries
@pytest.mark.parametrize("m", [5, 11, 22, 40, 46])
def test_kernels_equal_scalar_reference(name, m):
    row = _x(h=1, w=23)[:1, :1]
    got = resample(row, (1, m), name)  # the height is unchanged: one axis only
    ref = _reference_1d(row.flatten().tolist(), m, name)
    assert got.flatten().tolist() == pytest.approx(ref, abs=1e-12)


@pytest.mark.parametrize("name", sorted(KERNELS))
@pytest.mark.parametrize("size", SIZES)
def test_constant_image_stays_constant(name, size):
    out = resample(torch.full((1, 2, 37, 64), 0.37, dtype=torch.float64), size, name)
    assert out.shape[-2:] == size
    assert (out - 0.37).abs().max() < 1e-12


@pytest.mark.parametrize("name", sorted(KERNELS))
def test_unchanged_axis_is_not_resampled(name):
    x = _x()
    assert resample(x, (37, 64), name) is x  # also Mitchell, which does not interpolate
    wide = resample(x, (37, 20), name)  # width only: the rows stay independent
    rows = resample(x[:, :, :5], (5, 20), name)
    assert (wide[:, :, :5] - rows).abs().max() < 1e-14


def test_box_boundary_sample_belongs_to_one_window():
    # 5 -> 3: the windows are [-0.5, 1.17), [1.17, 2.83), [2.83, 4.5); torch's `area` differs here
    x = torch.zeros(1, 1, 1, 5, dtype=torch.float64)
    x[..., 2] = 1
    assert resample(x, (1, 3), "aa_box").flatten().tolist() == [0, 1, 0]
    # 7 -> 6: sample 3 sits exactly on the boundary of windows 2 and 3 and belongs to window 3
    # (a float offset rounds it into window 2: a half-pixel shift)
    x = torch.zeros(1, 1, 1, 7, dtype=torch.float64)
    x[..., 3] = 1
    assert resample(x, (1, 6), "aa_box").flatten().tolist() == [0, 0, 0, 0.5, 0, 0]
    # 3 -> 6 (upscale): nearest sample, each input twice
    ramp = torch.arange(3, dtype=torch.float64).view(1, 1, 1, 3)
    assert resample(ramp, (1, 6), "aa_box").flatten().tolist() == [0, 0, 1, 1, 2, 2]
    # 4 -> 2: every sample sits inside one window
    ramp = torch.arange(4, dtype=torch.float64).view(1, 1, 1, 4)
    assert resample(ramp, (1, 2), "aa_box").flatten().tolist() == [0.5, 2.5]


def test_scaled_size_is_interpolate_size():
    for h, w, f in [
        (96, 96, 0.5),
        (96, 80, 1.37),
        (7, 9, 1.3),
        (64, 61, 0.3),
        (5, 5, 1),
    ]:
        x = torch.zeros(1, 1, h, w)
        ref = F.interpolate(x, scale_factor=f, mode="bilinear").shape[-2:]
        assert scaled_size(h, w, f) == tuple(ref)
    # an empty size is an error, as in F.interpolate, not a silent 1 px image
    assert scaled_size(64, 64, 0.01) == (0, 0)
    with pytest.raises(ValueError, match="output size must be positive"):
        resample(torch.zeros(1, 1, 64, 64), (0, 0), "aa_bicubic")


@pytest.mark.parametrize("name", sorted(KERNELS))
def test_ambient_autocast_keeps_dtype_and_precision(name):
    x = _x().float()
    plain = resample(x, (18, 31), name)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cast = resample(x, (18, 31), name)
    assert cast.dtype == torch.float32
    assert torch.equal(cast, plain)


# ---------------------------------------------------------------- Degrader keys
def test_explicit_legacy_list_is_the_default():
    """Absent keys and the legacy list spelled out draw and compute the same (bitwise)."""
    outs = []
    for extra in ({}, {k: ["area", "bilinear", "bicubic"] for k in MODE_KEYS}):
        d = Degrader({**CFG, **extra}, scale=S, device="cpu")
        _seed(5)
        outs.append([d(*_batch(seed=i, dev="cpu")) for i in range(3)])
    for a, b in zip(*outs, strict=True):
        assert torch.equal(a, b)


def test_default_draws_as_before(monkeypatch):
    """The mode of an absent key comes from one random.choice over the legacy tuple, as before."""
    seen = []
    monkeypatch.setattr(
        od.random, "choice", lambda seq: seen.append(tuple(seq)) or seq[0]
    )
    monkeypatch.setattr(
        od.random,
        "choices",
        lambda seq, _weights: [seq[2]],  # resize kind "keep"
    )
    Degrader(CFG, scale=S, device="cpu")(*_batch(dev="cpu"))
    assert seen == [("area", "bilinear", "bicubic")] * 3


@pytest.mark.parametrize("order", [0.1, 0.9])  # both final orders
def test_antialiased_modes_reach_every_resize(monkeypatch, order):
    calls = []

    def spy(x, size, kernel):
        calls.append((tuple(x.shape[-2:]), tuple(size), kernel))
        return resample(x, size, kernel)

    monkeypatch.setattr(od, "resample", spy)
    monkeypatch.setattr(
        od, "rng", type("R", (), {"uniform": staticmethod(lambda *_: order)})()
    )
    monkeypatch.setattr(
        od.random,
        "choices",
        lambda seq, _weights: [seq[1]],  # resize kind "down"
    )
    cfg = {
        **CFG,
        "resize_mode": ["aa_lanczos3"],
        "resize_mode2": ["aa_mitchell"],
        "final_resize_mode": ["aa_bicubic"],
    }
    gt, *kernels = _batch(h=96, dev="cpu")
    lq = Degrader(cfg, scale=S, device="cpu")(gt, *kernels)
    # "down" with a stubbed uniform: factor = order for both stages
    first = scaled_size(96, 96, order)
    second = (int(96 / S * order), int(96 / S * order))
    assert calls == [
        ((96, 96), first, "aa_lanczos3"),
        (first, second, "aa_mitchell"),
        (second, (24, 24), "aa_bicubic"),
    ]
    assert lq.shape == (2, 3, 24, 24)
    assert lq.min() >= 0
    assert lq.max() <= 1
    assert torch.equal(lq, (lq * 255).round() / 255)  # still quantised to 8 bit


def test_mode_weights_select(monkeypatch):
    picked = []
    real = od.resample
    monkeypatch.setattr(
        od, "resample", lambda x, size, k: picked.append(k) or real(x, size, k)
    )
    cfg = {**CFG}
    for key, prob_key in MODE_KEYS.items():
        cfg[key] = ["bicubic", "aa_box", "aa_lanczos2"]
        cfg[prob_key] = [0, 0, 3]
    d = Degrader(cfg, scale=S, device="cpu")
    _seed(1)
    for i in range(4):
        d(*_batch(seed=i, dev="cpu"))
    assert picked
    assert set(picked) == {"aa_lanczos2"}


def _textured(b=2, h=96):
    """8 px blocks of random colour and identity kernels: the resize modes stay visible in the
    LQ (the random blur kernels of `_batch` flatten its noise image to a constant)."""
    g = torch.Generator().manual_seed(0)
    gt = torch.rand(b, 3, h // 8, h // 8, generator=g).repeat_interleave(8, 2)
    pulses = []
    for k in (7, 9, 21):
        pulse = torch.zeros(b, k, k)
        pulse[:, k // 2, k // 2] = 1
        pulses.append(pulse)
    return gt.repeat_interleave(8, 3), *pulses


def test_mixed_list_is_seeded_and_differs_from_legacy():
    cfg = {**CFG, **{k: ["aa_bicubic", "aa_lanczos2", "aa_box"] for k in MODE_KEYS}}
    outs = []
    for c in (cfg, cfg, CFG):
        d = Degrader(c, scale=S, device="cpu")
        _seed(3)
        outs.append(d(*_textured()))
    assert torch.equal(outs[0], outs[1])
    assert not torch.equal(outs[0], outs[2])


@pytest.mark.parametrize("key", MODE_KEYS)
@pytest.mark.parametrize(
    ("value", "prob", "match"),
    [
        (["lanczos"], None, "unknown mode 'lanczos'"),
        ([], None, "non-empty list"),
        ("aa_bicubic", None, "non-empty list"),
        ([3], None, "unknown mode 3"),
        (None, [1, 1, 1], "needs"),
        (["area", "aa_box"], [1], "must be 2 weights"),
        (["area", "aa_box"], [1, -1], "must be 2 weights"),
        (["area", "aa_box"], [0, 0], "must be 2 weights"),
        (["area", "aa_box"], [1, True], "must be 2 weights"),
        (["area", "aa_box"], [1, float("nan")], "must be 2 weights"),
        (["area", "aa_box"], "0.5", "must be 2 weights"),
    ],
)
def test_bad_mode_keys_raise(key, value, prob, match):
    cfg = {**CFG, key: value, MODE_KEYS[key]: prob}
    with pytest.raises(ConfigError, match=match) as err:
        Degrader(cfg, scale=S, device="cpu")
    named = key if prob is None else MODE_KEYS[key]
    assert re.match(rf"\[degradations\] {named}(?!\w)", str(err.value))


def test_unknown_mode_suggests_a_name():
    with pytest.raises(ConfigError, match=r"did you mean 'aa_lanczos[23]'"):
        Degrader({**CFG, "resize_mode": ["aa_lanczos"]}, scale=S, device="cpu")


def test_schema_knows_the_keys():
    keys = {*MODE_KEYS, *MODE_KEYS.values()}
    assert keys <= config.SCHEMA["degradations"].keys()
    assert keys <= config.SCHEMA["datasets.*"].keys()
    assert all(config.SCHEMA["degradations"][k] is list for k in keys)


def _states():
    return random.getstate(), rng_util.get_state(), torch.get_rng_state()


@pytest.mark.parametrize("cfg", [base.CFG, base.NO_JPEG], ids=["jpeg", "no_jpeg"])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_default_consumes_every_rng_like_the_frozen_reference(cfg, seed):
    """Outputs AND the three generator states (python, neosr/numpy, torch CPU) after default-config
    batches equal the frozen pre-refactor reference, with and without JPEG (design D6)."""
    d = Degrader(cfg, scale=S, device="cpu")
    batches = [_batch(seed=10 * seed + i, dev="cpu") for i in range(3)]
    _seed(seed)
    got = list(starmap(d, batches))
    after = _states()
    _seed(seed)
    jpeger = DiffJPEG(differentiable=False)
    ref = [base._reference(cfg, jpeger, *b) for b in batches]
    ref_after = _states()
    for g, r in zip(got, ref, strict=True):
        assert torch.equal(g, r)
    assert after[0] == ref_after[0]
    assert after[1] == ref_after[1]
    assert torch.equal(after[2], ref_after[2])


# ---------------------------------------------------------------- O2: gates and the clean path
GATES = ("noise_prob", "noise_prob2", "jpeg_prob", "jpeg_prob2")
NOISE_OFF = {"noise_prob": 0, "noise_prob2": 0}
JPEG_OFF = {"jpeg_prob": 0, "jpeg_prob2": 0}


def _lq(cfg, seed=7):
    """One textured batch of two under fixed seeds; with the stubbed mask the runs differ by the
    gates only (the stub draws nothing, so every stream stays aligned)."""
    d = Degrader(cfg, scale=S, device="cpu")
    _seed(seed)
    return d(*_textured())


@pytest.fixture
def first_row_only(monkeypatch):
    """Every gate lets sample 0 through and holds sample 1 back; records the probabilities."""
    asked = []

    def mask(x, p):
        asked.append(p)
        return torch.tensor([True, False]).view(2, 1, 1, 1).expand(x.size(0), 1, 1, 1)

    monkeypatch.setattr(Degrader, "_mask", staticmethod(mask))
    return asked


def test_gates_at_their_defaults_change_nothing():
    explicit = {**CFG, **dict.fromkeys(GATES, 1), "clean_prob": 0, "jpeg_prob2": 1.0}
    runs = []
    for cfg in (CFG, explicit):
        d = Degrader(cfg, scale=S, device="cpu")
        _seed(4)
        runs.append(([d(*_batch(seed=i, dev="cpu")) for i in range(3)], _states()))
    (out_a, st_a), (out_b, st_b) = runs
    for a, b in zip(out_a, out_b, strict=True):
        assert torch.equal(a, b)
    assert st_a[:2] == st_b[:2]
    assert torch.equal(st_a[2], st_b[2])


SEEDS = [0, 1, 2, 7]  # both final orders and several resize kinds


@pytest.mark.parametrize("seed", SEEDS)
def test_noise_gate_is_per_sample(first_row_only, seed):
    full = _lq(base.NO_JPEG, seed)
    none = _lq({**base.NO_JPEG, **NOISE_OFF}, seed)
    assert first_row_only == []  # gates at 0 and 1 never ask for a mask
    mixed = _lq({**base.NO_JPEG, "noise_prob": 0.25, "noise_prob2": 0.75}, seed)
    assert first_row_only == [0.25, 0.75]
    assert not torch.equal(full[1], none[1])
    assert torch.equal(mixed[0], full[0])
    assert torch.equal(mixed[1], none[1])


@pytest.mark.parametrize("seed", SEEDS)
def test_jpeg_gate_is_per_sample(first_row_only, seed):
    full = _lq({**CFG, **NOISE_OFF}, seed)
    none = _lq({**CFG, **NOISE_OFF, **JPEG_OFF}, seed)
    assert first_row_only == []
    mixed = _lq({**CFG, **NOISE_OFF, "jpeg_prob": 0.25, "jpeg_prob2": 0.75}, seed)
    assert first_row_only == [0.25, 0.75]
    assert not torch.equal(full[1], none[1])
    assert torch.equal(mixed[0], full[0])
    assert torch.equal(mixed[1], none[1])


def test_jpeg_gate_without_jpeg_draws_nothing(first_row_only):
    cfg = {**base.NO_JPEG, **NOISE_OFF, "jpeg_prob": 0.5, "jpeg_prob2": 0.5}
    d = Degrader(cfg, scale=S, device="cpu")
    _seed(1)
    before = torch.get_rng_state()
    d(*_textured())
    assert first_row_only == []
    assert torch.equal(
        torch.get_rng_state(), before
    )  # no noise, no quality, no mask draw


def test_noise_off_keeps_the_resize_draws_of_a_seed():
    """The Gaussian/Poisson coin is drawn at every noise_prob, so python and numpy streams, and
    with them the resize factors and modes of a seed, do not depend on the gate."""
    states = []
    for extra in ({}, NOISE_OFF):
        d = Degrader({**base.NO_JPEG, **extra}, scale=S, device="cpu")
        _seed(2)
        d(*_textured())
        states.append(_states())
    assert states[0][:2] == states[1][:2]


@pytest.mark.usefixtures("first_row_only")
def test_skipped_noise_is_clamped_like_noisy():
    x = torch.linspace(-0.5, 1.5, 2 * 3 * 4 * 4).view(2, 3, 4, 4)
    off = Degrader({**CFG, **NOISE_OFF}, scale=S, device="cpu")
    assert torch.equal(off._noise(x, ""), x.clamp(0, 1))
    half = Degrader({**CFG, "noise_prob": 0.5}, scale=S, device="cpu")
    _seed(0)
    out = half._noise(x, "")
    assert torch.equal(out[1], x[1].clamp(0, 1))
    assert not torch.equal(out[0], x[0].clamp(0, 1))
    assert out.min() >= 0
    assert out.max() <= 1


def test_mask_shape_rate_and_stream():
    torch.manual_seed(0)
    m = Degrader._mask(torch.zeros(4000, 3, 2, 2), 0.3)
    assert m.shape == (4000, 1, 1, 1)
    assert m.dtype == torch.bool
    assert 0.27 < m.float().mean() < 0.33  # 4 sigma of a binomial(4000, 0.3)
    torch.manual_seed(0)
    assert torch.equal(m, Degrader._mask(torch.zeros(4000, 1, 1, 1), 0.3))


def test_all_gates_ask_in_chain_order(first_row_only):
    cfg = {**CFG, "noise_prob": 0.1, "jpeg_prob": 0.2, "noise_prob2": 0.3}
    _lq({**cfg, "jpeg_prob2": 0.4, "clean_prob": 0.5})
    assert first_row_only == [0.1, 0.2, 0.3, 0.4, 0.5]


def _clean_ref(gt, mode="aa_bicubic"):
    size = (gt.shape[2] // S, gt.shape[3] // S)
    if mode in KERNELS:
        return od._quantize(resample(gt, size, mode))
    return od._quantize(F.interpolate(gt, size=size, mode=mode))


def test_clean_only_is_one_resize_and_no_chain(first_row_only):
    gt, *kernels = _textured()
    d = Degrader({**CFG, "clean_prob": 1}, scale=S, device="cpu")
    _seed(5)
    before = _states()
    lq = d(gt, *kernels)
    after = _states()
    assert torch.equal(lq, _clean_ref(gt))
    assert first_row_only == []
    assert after[1] == before[1]  # no chain: numpy untouched
    assert torch.equal(after[2], before[2])  # no noise, JPEG or mask draw


@pytest.mark.parametrize("seed", SEEDS)
def test_clean_gate_is_per_sample(first_row_only, seed):
    gt, *_ = _textured()
    chain = _lq(CFG, seed)
    assert first_row_only == []
    mixed = _lq({**CFG, "clean_prob": 0.5}, seed)
    assert first_row_only == [0.5]
    assert torch.equal(mixed[0], _clean_ref(gt)[0])
    assert torch.equal(mixed[1], chain[1])
    assert not torch.equal(mixed[0], chain[0])


@pytest.mark.parametrize(
    ("modes", "weights", "expect"),
    [
        (["area"], None, "area"),  # a legacy mode: F.interpolate
        (["aa_lanczos3"], None, "aa_lanczos3"),
        (["aa_lanczos3", "aa_box"], [0, 1], "aa_box"),
    ],
)
def test_clean_mode_list(modes, weights, expect):
    gt, *kernels = _textured()
    cfg = {**CFG, "clean_prob": 1, "clean_mode": modes, "clean_mode_prob": weights}
    assert torch.equal(
        Degrader(cfg, scale=S, device="cpu")(gt, *kernels), _clean_ref(gt, expect)
    )


def test_clean_with_luma_requantises_after_the_rgb_quantisation():
    gt, *kernels = _textured()
    d = Degrader({**CFG, "clean_prob": 1, "color": "y"}, scale=S, device="cpu")
    gy, ly = d.degrade(gt, *kernels)
    w = torch.tensor(od._LUMA).view(1, 3, 1, 1)
    assert gy.shape == (2, 1, 96, 96)
    assert torch.equal(ly, od._quantize((_clean_ref(gt) * w).sum(1, keepdim=True)))
    assert torch.equal(
        (ly * 255).round() / 255, ly
    )  # what the pair pool stores as uint8


@pytest.mark.parametrize("key", [*GATES, "clean_prob"])
@pytest.mark.parametrize("value", [-0.1, 1.5, True, "0.5", [0.5], float("nan")])
def test_bad_probability_raises(key, value):
    with pytest.raises(
        ConfigError, match=rf"\[degradations\] {key} must be a probability"
    ):
        Degrader({**CFG, key: value}, scale=S, device="cpu")


def test_bad_clean_mode_raises():
    with pytest.raises(ConfigError, match=r"\[degradations\] clean_mode: unknown mode"):
        Degrader({**CFG, "clean_mode": ["clean"]}, scale=S, device="cpu")
    with pytest.raises(ConfigError, match=r"\[degradations\] clean_mode_prob needs"):
        Degrader({**CFG, "clean_mode_prob": [1]}, scale=S, device="cpu")


def test_schema_knows_the_o2_keys():
    keys = {*GATES, "blur_prob", "clean_prob", "clean_mode", "clean_mode_prob"}
    for section in ("degradations", "datasets.*"):
        assert keys <= config.SCHEMA[section].keys()
    assert all(config.SCHEMA["degradations"][k] is float for k in (*GATES, "blur_prob"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_no_host_sync_with_every_option_on():
    cfg = {**CFG, **dict.fromkeys(GATES, 0.5), "clean_prob": 0.5}
    cfg |= {k: ["aa_bicubic", "aa_lanczos2", "aa_box", "bicubic"] for k in MODE_KEYS}
    d = Degrader(cfg, scale=4, device="cuda")
    gt = torch.rand(4, 3, 96, 96, device="cuda")
    ks = [torch.full((4, n, n), 1 / n**2, device="cuda") for n in (7, 9, 21)]
    d(gt, *ks)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for i in range(20):
            _seed(i)
            d(gt, *ks)
    finally:
        torch.cuda.set_sync_debug_mode(0)


# ---------------------------------------------------------------- O2: blur_prob in the worker
def _worker_items(root, n=12, **kw):
    ds = build_dataset(data._otf_opt(root, **kw))
    items = data._items(ds, n)  # seeds python, numpy and torch with 3, then loads
    return ds, items, _states()


def _is_pulse(ds, kernel):
    return torch.equal(kernel, ds.pulse_tensor)


def test_blur_prob_one_and_none_are_the_default(pairs):
    _, ref, ref_state = _worker_items(pairs)
    for p in (1, 1.0, None):
        _, got, state = _worker_items(pairs, blur_prob=p)
        data._assert_items_equal(got, ref)
        assert state[:2] == ref_state[:2]
        assert torch.equal(state[2], ref_state[2])


class _CountingRng:
    """neosr's rng with its argument-free coins counted (the probability draws of the worker)."""

    def __init__(self, real):
        self.real, self.coins = real, 0

    def uniform(self, *args):
        self.coins += not args
        return self.real.uniform(*args)

    def __getattr__(self, name):
        return getattr(self.real, name)


@pytest.mark.parametrize(("blur_prob", "coins"), [(None, 3), (1, 3), (0, 3), (0.5, 4)])
def test_worker_coin_count_per_sample(pairs, monkeypatch, blur_prob, coins):
    """The worker before `blur_prob` drew three argument-free coins per sample (sinc_prob,
    sinc_prob2, final_sinc_prob). The default and the endpoints must not add one: a draw there
    would shift every later kernel of a seeded run; in between the gate adds exactly one."""
    counting = _CountingRng(od.rng)
    monkeypatch.setattr(data.otf_dataset, "rng", counting)
    ds = build_dataset(data._otf_opt(pairs, blur_prob=blur_prob))
    items = data._items(ds, 10)
    assert counting.coins == coins * 10
    for item in items:
        assert item["kernel1"].dtype == torch.float32
        assert item["kernel1"].shape == (21, 21)


def test_blur_prob_zero_skips_every_first_blur_without_a_draw(pairs):
    _, ref, ref_state = _worker_items(pairs)
    ds, got, state = _worker_items(pairs, blur_prob=0)
    assert all(_is_pulse(ds, item["kernel1"]) for item in got)
    assert not any(_is_pulse(ds, item["kernel1"]) for item in ref)
    for a, b in zip(got, ref, strict=True):  # everything else is the same sample
        assert torch.equal(a["gt"], b["gt"])
        assert torch.equal(a["kernel2"], b["kernel2"])
        assert torch.equal(a["sinc_kernel"], b["sinc_kernel"])
    assert state[:2] == ref_state[:2]
    assert torch.equal(state[2], ref_state[2])


def test_blur_prob_between_draws_one_coin_per_sample(pairs):
    ds, got, state = _worker_items(pairs, n=200, blur_prob=0.5)
    _, _, ref_state = _worker_items(pairs, n=200)
    skipped = sum(_is_pulse(ds, item["kernel1"]) for item in got)
    assert 72 <= skipped <= 128  # 4 sigma of a binomial(200, 0.5)
    assert state[1] != ref_state[1]  # the coin comes from neosr's numpy stream


def test_a_skipped_first_blur_is_the_identity():
    gt = _x(torch.float32, 32, 32)
    pulse = torch.zeros(2, 21, 21)
    pulse[:, 10, 10] = 1
    assert torch.equal(filter2D(gt, pulse), gt)


@pytest.mark.parametrize("value", [-0.1, 1.5, True, "0.5", [1]])
def test_bad_blur_prob_raises(pairs, value):
    with pytest.raises(
        ConfigError, match=r"\[degradations\] blur_prob must be a probability"
    ):
        build_dataset(data._otf_opt(pairs, blur_prob=value))
