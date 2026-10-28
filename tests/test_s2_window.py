"""Stage 2, window-attention lane (W-01..W-15): exact SDPA path, masks, padding, checkpointing.

Parity is checked against the stage-2 base implementation, reconstructed from git (`BASE`): same
weights (strict load), outputs and every parameter gradient. On CPU the comparison is against the fp32
noise floor: both fp32 runs are measured against a float64 run of the old code, and the new error must
stay within a small multiple of the old one.
"""

import copy
import inspect
import math
import subprocess
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from neosr.archs import arch_util, build_network
from neosr.archs import (  # noqa: F401  (registration-free: imported for the module objects)
    atd_arch,
    craft_arch,
    dat_arch,
    drct_arch,
    hat_arch,
    hitsrf_arch,
    rgt_arch,
    srformer_arch,
    swinir_arch,
    swinirdense_arch,
)
from neosr.archs.arch_util import shift_mask, window_sdpa
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[1]
BASE = "f3d2b7c"  # stage-2 base: the implementation every rewrite here must reproduce
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
NEW = {m.__name__.rsplit(".", 1)[1]: m for m in (
    atd_arch, craft_arch, dat_arch, drct_arch, hat_arch, hitsrf_arch, rgt_arch, srformer_arch,
    swinir_arch, swinirdense_arch,
)}  # fmt: skip
_OLD: dict[str, types.ModuleType] = {}
# dtype-only edits of the reference so it can run in float64 (the old HiT-SRF casts its table to fp32)
_OLD_PATCH = {"hitsrf_arch": [(".contiguous().float()", ".contiguous().to(v.dtype)")]}


def old(stem: str) -> types.ModuleType:
    if stem not in _OLD:
        src = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{BASE}:neosr/archs/{stem}.py"],
            capture_output=True, text=True, check=False,
        )  # fmt: skip
        if src.returncode:
            pytest.skip(f"base commit {BASE} not available")
        code = src.stdout
        for a, b in _OLD_PATCH.get(stem, ()):
            assert a in code
            code = code.replace(a, b)
        mod = types.ModuleType(f"_old_{stem}")
        mod.__package__ = "neosr.archs"  # relative imports of the old file
        exec(compile(code, f"{BASE}:{stem}.py", "exec"), mod.__dict__)  # noqa: S102
        _OLD[stem] = mod
    return _OLD[stem]


# reduced configs: every code path (shifted/unshifted, buffer/rebuilt mask, padding) at CPU cost
SMALL = {
    "swinir": ("swinir_arch", dict(img_size=16, window_size=4, depths=(2, 2), num_heads=(2, 2),
                                   embed_dim=24, upsampler="pixelshuffledirect")),
    "swinirdense": ("swinirdense_arch", dict(img_size=16, window_size=4, depths=(2, 2),
                                             num_heads=(2, 2), embed_dim=24,
                                             upsampler="pixelshuffledirect")),
    "hat": ("hat_arch", dict(img_size=16, window_size=4, depths=(2, 2), num_heads=(2, 2),
                             embed_dim=24, squeeze_factor=4, upsampler="pixelshuffle")),
    "drct": ("drct_arch", dict(img_size=16, window_size=4, depths=(2, 2), num_heads=(2, 2),
                               embed_dim=24, gc=8)),
    "dat": ("dat_arch", dict(img_size=16, split_size=[2, 4], depth=[2, 2], num_heads=[4, 4],
                             embed_dim=64, expansion_factor=2.0)),
    "rgt": ("rgt_arch", dict(img_size=16, split_size=[4, 8], depth=[2, 2], num_heads=[4, 4],
                             embed_dim=64, mlp_ratio=2)),
    "atd": ("atd_arch", dict(img_size=16, window_size=4, depths=(2, 2), num_heads=(2, 2),
                             embed_dim=24, category_size=512, num_tokens=8, reducted_dim=4,
                             convffn_kernel_size=3, mlp_ratio=1.0, upsampler="pixelshuffledirect")),
    "srformer": ("srformer_arch", dict(img_size=16, window_size=8, depths=(2, 2), num_heads=(2, 2),
                                       embed_dim=24, upsampler="pixelshuffledirect")),
    "craft": ("craft_arch", dict(window_size=16, embed_dim=96, depths=[1, 1], num_heads=[4, 4])),
    "craft_dead_pos": ("craft_arch", dict(window_size=16, embed_dim=48, depths=[1, 1],
                                          num_heads=[2, 2])),
    "hit_srf": ("hitsrf_arch", dict(img_size=16, base_win_size=[4, 4], depths=(2,), num_heads=(2,),
                                    embed_dim=24, hier_win_ratios=[0.5, 1, 2, 4])),
}  # fmt: skip
CLS = {"craft_dead_pos": "craft"}
# aligned to img_size (persistent-mask path), non-aligned (padded, mask rebuilt), odd
SIZES = {"craft": ((16, 16), (32, 16), (20, 12)), "craft_dead_pos": ((16, 16), (20, 12))}


def no_drop_path(cls) -> dict:
    return {"drop_path_rate": 0.0} if "drop_path_rate" in inspect.signature(cls).parameters else {}


def pair(name: str, new_kw: dict | None = None, **kw):
    stem, cfg = SMALL[name]
    cls = CLS.get(name, name)
    kw = {**no_drop_path(getattr(old(stem), cls)), **kw}
    torch.manual_seed(0)
    a = getattr(old(stem), cls)(upscale=2, **cfg, **kw)
    with torch.no_grad():  # make the learned bias matter (trained tables are far from N(0, 0.02))
        for n, p in a.named_parameters():
            if n.endswith("relative_position_bias_table"):
                p.normal_()
    b = getattr(NEW[stem], cls)(upscale=2, **cfg, **kw, **(new_kw or {}))
    b.load_state_dict(a.state_dict(), strict=True)  # old weights keep loading
    return a, b


def run(net, x, train=True):
    net.train(train).zero_grad(set_to_none=True)
    y = net(x)
    y.square().mean().backward()
    grads = {n: p.grad.double() for n, p in net.named_parameters() if p.grad is not None}
    return y.detach().double(), grads


def nrel(a, ref):
    return ((a - ref).norm() / ref.norm()).item()


def rel(a, ref, floor=1e-30):
    return ((a - ref).abs().max() / ref.abs().max().clamp_min(floor)).item()


def assert_noise_floor(a, b, x, k=4.0):
    """new fp32 within k x the old fp32 rounding error (both against old float64)."""
    r = copy.deepcopy(a).double()
    yr, gr = run(r, x.double())
    y0, g0 = run(a, x)
    y1, g1 = run(b, x)
    assert y1.shape == yr.shape
    assert set(g0) == set(gr)
    assert rel(y1, yr) <= k * rel(y0, yr) + 1e-6
    # gradients that are structurally ~0 (e.g. a per-head constant bias, softmax-invariant) are measured
    # against the overall gradient scale, not against their own float64 rounding noise
    floor = 1e-4 * max(g.abs().max().item() for g in gr.values())
    frozen = set(gr) - set(g1)  # CRAFT's dead position bias (W-08): its exact gradient is zero
    assert frozen == {n for n, p in b.named_parameters() if not p.requires_grad}
    for n in frozen:
        assert gr[n].abs().max() <= 1e-6 * floor, n
    for n in g1:
        assert rel(g1[n], gr[n], floor) <= k * rel(g0[n], gr[n], floor) + 1e-5, n


@pytest.fixture
def _fp32_exact():
    flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # cudnn: on by default
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags


# ---------------------------------------------------------------- W-01: the helper itself
def manual(q, k, v, bias, mask, scale):
    attn = q @ k.transpose(-2, -1) * scale
    if bias is not None:
        attn = attn + bias
    if mask is not None:
        nw = mask.shape[0]
        attn = (attn.view(-1, nw, *attn.shape[1:]) + mask.unsqueeze(1)).flatten(0, 1)
    return attn.softmax(-1) @ v


@pytest.mark.parametrize(
    ("d", "n", "m", "with_bias", "with_mask"),
    [(10, 16, 16, True, True), (8, 16, 16, True, False), (12, 16, 4, True, True),
     (5, 9, 9, False, True), (35, 16, 16, False, False)],
)  # fmt: skip
def test_window_sdpa_matches_manual(d, n, m, with_bias, with_mask):
    torch.manual_seed(0)
    b, nw, nh = 2, 3, 2
    q = torch.randn(b * nw, nh, n, d, dtype=torch.float64, requires_grad=True)
    k = torch.randn(b * nw, nh, m, d, dtype=torch.float64, requires_grad=True)
    v = torch.randn(b * nw, nh, m, d, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(nh, n, m, dtype=torch.float64, requires_grad=True) if with_bias else None
    mask = (torch.rand(nw, n, m) > 0.6).double() * -100 if with_mask else None
    leaves = [t for t in (q, k, v, bias) if t is not None]
    ref = manual(q, k, v, bias, mask, 0.3)
    g_ref = torch.autograd.grad(ref.square().sum(), leaves)
    out = window_sdpa(q, k, v, bias, mask, scale=0.3)
    g = torch.autograd.grad(out.square().sum(), leaves)
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, rtol=1e-10, atol=1e-12)
    for a, r in zip(g, g_ref, strict=True):
        torch.testing.assert_close(a, r, rtol=1e-10, atol=1e-12)


@cuda
@pytest.mark.usefixtures("_fp32_exact")
@pytest.mark.parametrize("with_mask", [False, True])
@pytest.mark.parametrize("d", [8, 10, 24, 30, 35, 53])  # the default head dims
def test_window_sdpa_cuda_fused_kernel(d, with_mask):
    # fp32 and bf16 on the memory-efficient kernel (the context raises if SDPA would fall back to the
    # math path), with SwinIR's non-contiguous gathered bias: against float64, no worse than the manual path
    torch.manual_seed(0)
    b, nw, nh, n = 2, 4, 3, 64
    q, k, v = (torch.randn(b * nw, nh, n, d, device="cuda", dtype=torch.float64) for _ in range(3))
    table = torch.randn(n * n, nh, device="cuda", dtype=torch.float64)
    mask = (torch.rand(nw, n, n, device="cuda") > 0.6).double() * -100 if with_mask else None
    go = torch.randn(b * nw, nh, n, d, device="cuda", dtype=torch.float64)

    def grads(fn, dt, cast):
        leaves = [t.to(dt).requires_grad_() for t in (q, k, v, table)]
        bias = leaves[3].view(n, n, nh).permute(2, 0, 1)  # as gathered in SwinIR: not contiguous
        out = fn(*(t.to(cast) for t in leaves[:3]), bias.to(cast), None if mask is None else mask.to(cast))
        return [t.double() for t in (out, *torch.autograd.grad(out, leaves, go.to(out.dtype)))]

    ref = grads(lambda q, k, v, b_, m: manual(q, k, v, b_, m, 0.3), torch.float64, torch.float64)
    for cast, tol in ((torch.float32, 1e-7), (torch.bfloat16, 1e-4)):
        base = grads(lambda q, k, v, b_, m: manual(q, k, v, b_, m, 0.3), torch.float32, cast)
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            new = grads(lambda q, k, v, b_, m: window_sdpa(q, k, v, b_, m, scale=0.3), torch.float32, cast)
        for t, t0, r in zip(new, base, ref, strict=True):
            assert nrel(t, r) <= 4 * nrel(t0, r) + tol, cast


def test_shift_mask_memo():
    calls = []

    def build(*args):
        calls.append(args)
        return torch.zeros(1)

    a = shift_mask(build, 8, 8, "cpu")
    assert shift_mask(build, 8, 8, "cpu") is a
    assert len(calls) == 1
    b = shift_mask(build, 16, 8, "cpu")  # another size replaces the single entry
    assert b is not a
    assert shift_mask(build, 8, 8, "cpu") is not a
    assert len(calls) == 3
    with torch.inference_mode():  # never handed to autograd outside inference mode
        c = shift_mask(build, 8, 8, "cpu")
    assert c.is_inference()
    assert not shift_mask(build, 8, 8, "cpu").is_inference()


# ---------------------------------------------------------------- whole-net parity (CPU)
@pytest.mark.parametrize("name", list(SMALL))
def test_parity_with_base(name):
    a, b = pair(name)
    for s in SIZES.get(name, ((16, 16), (12, 12), (13, 7))):
        torch.manual_seed(1)
        assert_noise_floor(a, b, torch.rand(2, 3, *s))


@pytest.mark.parametrize("name", [n for n in SMALL if n not in {"hit_srf", "craft_dead_pos"}])
def test_manual_path_matches_base(name):
    # flash_attn=False is the old attention: the other rewrites (masks on device, memo, reshapes,
    # padding, index_reverse) are bit-exact (CRAFT: rounding, its dead constant bias is skipped)
    a, b = pair(name, {"flash_attn": False})
    for s in SIZES.get(name, ((16, 16), (12, 12), (13, 7))):
        torch.manual_seed(1)
        x = torch.rand(2, 3, *s)
        y0, g0 = run(a, x)
        y1, g1 = run(b, x)
        tol = {"rtol": 0, "atol": 0} if name != "craft" else {"rtol": 1e-5, "atol": 1e-6}
        torch.testing.assert_close(y1, y0, **tol)
        for n in g1:
            torch.testing.assert_close(g1[n], g0[n], **tol)


# default rows: strict load of base weights (state_dict keys and shapes unchanged)
ROWS = {
    "swinir_small": "swinir_arch", "swinirdense": "swinirdense_arch", "hat_s": "hat_arch",
    "drct_s": "drct_arch", "dat_s": "dat_arch", "rgt_s": "rgt_arch", "atd_light": "atd_arch",
    "srformer_light": "srformer_arch", "craft": "craft_arch", "hit_srf": "hitsrf_arch",
}  # fmt: skip


@pytest.mark.parametrize("row", list(ROWS))
def test_default_rows_load_base_weights(row):
    a = getattr(old(ROWS[row]), row)(upscale=4)
    b = build_network({"type": row}, scale=4)
    b.load_state_dict(a.state_dict(), strict=True)
    assert "mean" not in b.state_dict()


# ---------------------------------------------------------------- W-01 on CUDA, default rows
CUDA_ROWS = ["swinir_small", "swinirdense", "hat_s", "drct_s", "dat_s", "rgt_s", "atd_light",
             "srformer_light", "craft"]  # fmt: skip


NO_DROP_PATH = {"atd_light", "craft"}  # rows without a drop_path_rate kwarg


@cuda
@pytest.mark.usefixtures("_fp32_exact")
@pytest.mark.parametrize("row", CUDA_ROWS)
def test_cuda_default_rows_parity(row):
    kw = {} if row in NO_DROP_PATH else {"drop_path_rate": 0.0}
    torch.manual_seed(0)
    a = getattr(old(ROWS[row]), row)(upscale=2, **kw)
    with torch.no_grad():
        for n, p in a.named_parameters():
            if n.endswith("relative_position_bias_table"):
                p.normal_()
    b = build_network({"type": row, **kw}, scale=2)
    b.load_state_dict(a.state_dict(), strict=True)
    a, b = a.cuda(), b.cuda()
    torch.manual_seed(1)
    x = torch.rand(2, 3, 32, 40, device="cuda")  # non-aligned: masks rebuilt, padded
    # Every trainable parameter must get a gradient on both paths (only CRAFT's frozen dead position bias is
    # exempt, W-08). Per parameter, an error is measured against the float64 run of the old code and scaled by
    # max(|g_ref|, 1e-2 * the largest |g_ref|): DAT/RGT position-bias output biases have an exactly-zero true
    # gradient (softmax shift invariance), so a pure relative error is meaningless there. Slack, from measured
    # worst cases: fp32 3e-3 (swinirdense: SDPA rounding flips one LeakyReLU at a 6.9e-8 pre-activation in
    # layers.1, moving upstream grads by up to 1.3e-3 on this scale; no other row exceeds 3e-7), bf16 5e-2
    # (dat_s 1.6e-2: bf16 grads are 1e-3..1e-1 off for both paths). The kernel test above is the tight one.
    yr, gr = run(copy.deepcopy(a).double(), x.double())
    y0, g0 = run(a, x)
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]):  # no math fallback
        y1, g1 = run(b, x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            yb, gb = run(b, x)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ya, ga = run(a, x)
    frozen = {n for n, p in b.named_parameters() if not p.requires_grad}
    assert all(".pos." in n for n in frozen)
    assert row == "craft" or not frozen
    train = {n for n, _ in b.named_parameters()} - frozen
    assert set(g1) == set(gb) == train
    assert train <= set(g0) & set(ga) & set(gr)
    assert nrel(y1, yr) <= 4 * nrel(y0, yr) + 1e-6
    # bf16 autocast: the new path stays within rounding (no further from fp32 than ~the old bf16 path)
    assert nrel(yb, yr) <= 1.5 * nrel(ya, yr) + 1e-3
    floor = 1e-2 * max(gr[n].norm().item() for n in train)
    for n in sorted(train):
        ref = max(gr[n].norm().item(), floor)
        e1, e0, eb, ea = ((g[n] - gr[n]).norm().item() / ref for g in (g1, g0, gb, ga))
        assert e1 <= 4 * e0 + 3e-3, n
        assert eb <= 1.5 * ea + 5e-2, n


# ---------------------------------------------------------------- W-02 / W-08
@pytest.mark.parametrize("row", ["swinir_small", "swinirdense"])
def test_flash_attn_trains_bias_tables(row):
    torch.manual_seed(0)
    net = build_network({"type": row, "flash_attn": True, "drop_path_rate": 0.0}, scale=2)
    net.train()(torch.rand(1, 3, 20, 24)).mean().backward()
    tables = [p.grad for n, p in net.named_parameters() if n.endswith("relative_position_bias_table")]
    assert tables
    assert all(g is not None and g.abs().sum() > 0 for g in tables)


def test_craft_dead_position_bias_frozen():
    net = build_network({"type": "craft"}, scale=2)
    pos = [p for n, p in net.named_parameters() if ".attns." in n and ".pos." in n]
    assert len(pos) == 448  # 16 blocks x 2 branches x 14
    assert not any(p.requires_grad for p in pos)  # dead (pos_dim 1): frozen, still in state_dict
    net.train()(torch.rand(2, 3, 16, 16)).mean().backward()
    assert all(p.grad is not None for p in net.parameters() if p.requires_grad)  # DDP-safe
    live = build_network({"type": "craft", "embed_dim": 96}, scale=2)  # pos_dim 3: trained
    assert all(p.requires_grad for n, p in live.named_parameters() if ".pos." in n)


# ---------------------------------------------------------------- W-03 HiT-SRF pooled bias
@pytest.mark.parametrize("win", [4, 8, 16, 48])
def test_hitsrf_pooled_bias_matches_dense(win):
    torch.manual_seed(0)
    kw = dict(dim=24, base_win_size=(8, 8), window_size=(win, win), num_heads=2)
    a = old("hitsrf_arch").SCC(**kw).double()
    b = hitsrf_arch.SCC(**kw).double()
    b.load_state_dict(a.state_dict(), strict=True)
    x = torch.rand(2, win, win, 24, dtype=torch.float64)
    ya, yb = a(x), b(x)
    torch.testing.assert_close(yb, ya, rtol=1e-10, atol=1e-12)
    ga = torch.autograd.grad(ya.square().sum(), list(a.parameters()))
    gb = torch.autograd.grad(yb.square().sum(), list(b.parameters()))
    for p, q in zip(gb, ga, strict=True):
        torch.testing.assert_close(p, q, rtol=1e-9, atol=1e-12)


@cuda
def test_hitsrf_medium_trains():
    net = build_network({"type": "hit_srf_medium"}, scale=2).cuda().train()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = net(torch.rand(1, 3, 32, 32, device="cuda"))
    y.float().mean().backward()
    assert y.shape == (1, 3, 64, 64)
    assert torch.cuda.max_memory_allocated() < 4 * 2**30  # was > 20 GiB (dense position bias)


# ---------------------------------------------------------------- W-04 no host->device copies
def test_index_reverse_is_inverse_permutation():
    idx = torch.stack([torch.randperm(97) for _ in range(5)])
    rev = atd_arch.index_reverse(idx)
    torch.testing.assert_close(torch.gather(idx, -1, rev), torch.arange(97).expand(5, -1))


@cuda
@pytest.mark.parametrize("row", ["atd_light", "srformer_light", "rgt_s"])
def test_forward_has_no_blocking_copies(row):
    net = build_network({"type": row}, scale=2).cuda().train()
    x = torch.rand(1, 3, 40, 40, device="cuda")  # not img_size: masks are rebuilt
    net(x)  # warm-up (lazy init)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        net(x)
    finally:
        torch.cuda.set_sync_debug_mode("default")


# ---------------------------------------------------------------- W-05 one mask per forward
@pytest.mark.parametrize(
    ("stem", "factory", "builder"),
    [("swinir_arch", "swinir", swinir_arch), ("drct_arch", "drct", swinir_arch),
     ("srformer_arch", "srformer", srformer_arch), ("dat_arch", "dat", dat_arch),
     ("rgt_arch", "rgt", dat_arch)],
)  # fmt: skip
def test_one_shift_mask_per_forward(monkeypatch, stem, factory, builder):
    built = []
    real = builder.calculate_mask

    def spy(*args):
        built.append(args)
        return real(*args)

    for mod in {builder, NEW[stem]}:
        monkeypatch.setattr(mod, "calculate_mask", spy)
    arch_util._MEMO.clear()
    net = getattr(NEW[stem], factory)(upscale=2, **SMALL[{"swinir_arch": "swinir"}.get(stem, factory)][1])
    built.clear()  # construction builds the img_size buffers
    masks = []
    for m in net.modules():
        if type(m).__name__ in {"WindowAttention", "PSA", "Spatial_Attention"}:
            m.register_forward_pre_hook(
                lambda mod, args, kw: masks.append(kw.get("mask", args[-1] if len(args) > 3 else None)),
                with_kwargs=True,
            )
    with torch.no_grad():
        net.eval()(torch.rand(1, 3, 24, 24))  # not img_size
        assert len(built) == 1
        net(torch.rand(1, 3, 24, 24))  # next step: memo hit
        assert len(built) == 1
    assert None in masks  # unshifted blocks get no (all-zero) mask
    assert all(m is None or m.abs().max() == 100 for m in masks)


# ---------------------------------------------------------------- W-06 CRAFT padding
def test_craft_pads_to_window_multiple():
    seen = []
    for extra in (False, True):
        net = build_network({"type": "craft", "pad_extra_window": extra}, scale=2).eval()
        net.conv_first.register_forward_hook(lambda m, i, o: seen.append(tuple(i[0].shape[2:])))
        with torch.no_grad():
            for s in ((64, 64), (20, 12)):
                assert net(torch.rand(1, 3, *s)).shape == (1, 3, 2 * s[0], 2 * s[1])
    assert seen == [(64, 64), (32, 16), (80, 80), (32, 16)]


# ---------------------------------------------------------------- W-07 / W-11 checkpointing
@pytest.mark.parametrize(
    ("name", "flag"),
    [("srformer", "use_checkpoint"), ("hat", "use_checkpoint"), ("drct", "use_checkpoint"),
     ("atd", "use_checkpoint"), ("swinir", "use_checkpoint")],
)  # fmt: skip
def test_checkpoint_grad_parity(name, flag):
    stem, cfg = SMALL[name]
    x = torch.rand(2, 3, 13, 11)
    grads = []
    for on in (False, True):
        torch.manual_seed(0)
        net = getattr(NEW[stem], name)(upscale=2, **cfg, **{flag: on}).train()
        params = [p for p in net.parameters() if p.requires_grad]
        torch.manual_seed(1)  # same DropPath draws: non-reentrant checkpoint replays the RNG
        net(x).square().mean().backward(inputs=params)  # the trainer's call
        grads.append([p.grad for p in params])
    for g0, g1 in zip(*grads, strict=True):
        torch.testing.assert_close(g1, g0, rtol=1e-5, atol=1e-7)


def test_no_checkpointing_in_eval(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("checkpoint called in eval")

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", boom)
    for name in ("swinir", "swinirdense", "srformer", "dat", "rgt", "hit_srf"):
        stem, cfg = SMALL[name]
        flag = "use_chk" if name in {"dat", "rgt"} else "use_checkpoint"
        net = getattr(NEW[stem], name)(upscale=2, **cfg, **{flag: True}).eval()
        with torch.no_grad():
            net(torch.rand(1, 3, 16, 16))


# ---------------------------------------------------------------- W-10 tiny inputs
def small(name: str, **kw):
    stem, cfg = SMALL[name]
    return getattr(NEW[stem], CLS.get(name, name))(upscale=2, **{**cfg, **kw})


# reduced configs, same failure modes: reflect pad >= size (SRFormer), one mirror step too short (ATD, CRAFT:
# window 8 / 16 > 2x the input), log4(H // 16) of a side < 16 (RGT, eval) or log4(H // 4) < 4 (RGT, train)
@pytest.mark.parametrize(("name", "kw"), [("srformer", {}), ("atd", {"window_size": 8}), ("craft", {}),
                                          ("rgt", {})])  # fmt: skip
def test_tiny_inputs(name, kw):
    net = small(name, **kw)
    with torch.no_grad():
        for train, sizes in ((False, ((1, 1), (5, 7), (7, 5), (12, 12), (17, 9))), (True, ((2, 3),))):
            net.train(train)
            for h, w in sizes:
                assert net(torch.rand(1, 3, h, w)).shape == (1, 3, 2 * h, 2 * w)


def test_rgt_small_training_inputs_train_every_parameter():
    # sizes that used to raise: padded so the recursion depth is >= 1 (depth 0 leaves reduction1 with grad
    # None, which breaks DDP); 4..15 on both sides keeps its old depth 0 (exact, not covered here)
    net = small("rgt").train()
    for h, w in ((2, 3), (3, 8), (2, 20), (17, 1)):
        net.zero_grad(set_to_none=True)
        y = net(torch.rand(1, 3, h, w))
        assert y.shape == (1, 3, 2 * h, 2 * w)
        y.mean().backward()
        assert all(p.grad is not None for p in net.parameters())


def test_atd_fp16_scaled_query_does_not_overflow():
    # fp16 keeps the manual order (q @ k^T, then * scale): q * scale (100) overflows for |q| > 655
    torch.manual_seed(0)
    kw = {"dim": 8, "input_resolution": (4, 4), "num_tokens": 4, "num_heads": 2, "category_size": 8}
    msa = {f: atd_arch.AC_MSA(**kw, flash_attn=f) for f in (False, True)}
    msa[True].load_state_dict(msa[False].state_dict())
    with torch.no_grad():
        msa[False].logit_scale.fill_(math.log(100.0))
        msa[True].logit_scale.fill_(math.log(100.0))
    qkv = torch.randn(1, 16, 24) * 0.01
    qkv[..., :8] = 1000.0  # large queries, small keys: q @ k^T * scale is finite
    sim = torch.randn(1, 16, 4)
    out = {}
    for f, m in msa.items():
        m.half()
        x = qkv.half().requires_grad_()
        y = m(x, sim, (4, 4))
        y.float().square().sum().backward()
        out[f] = (y, x.grad, m.logit_scale.grad)
        assert all(torch.isfinite(t).all() for t in out[f])
    for a, b in zip(out[True], out[False], strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


# ---------------------------------------------------------------- W-13 / W-14 / W-15
def test_dat_rgt_manual_softmax_not_pinned(monkeypatch):
    seen = []
    real = F.softmax

    def spy(x, *a, **k):
        seen.append(k.get("dtype"))
        return real(x, *a, **k)

    monkeypatch.setattr(torch.nn.functional, "softmax", spy)
    for name in ("dat", "rgt"):
        stem, cfg = SMALL[name]
        net = getattr(NEW[stem], name)(upscale=2, **cfg, flash_attn=False).train()
        net(torch.rand(2, 3, 16, 16))
    assert seen
    assert all(d is None for d in seen)  # autocast may run it in fp32


@pytest.mark.parametrize("name", ["swinir", "swinirdense", "hat", "drct"])
def test_trace_is_not_frozen_to_the_sample_size(name):
    stem, cfg = SMALL[name]
    torch.manual_seed(0)
    net = getattr(NEW[stem], name)(upscale=2, **cfg).eval()
    with torch.no_grad():
        traced = torch.jit.trace(net, torch.rand(1, 3, 16, 16), check_trace=False)  # = img_size
        x = torch.rand(1, 3, 13, 18)  # padded, mask rebuilt
        torch.testing.assert_close(traced(x), net(x), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", ["srformer", "atd", "craft", "rgt"])
def test_mean_is_a_non_persistent_buffer(name):
    net = small(name)
    assert "mean" in dict(net.named_buffers())
    assert "mean" not in net.state_dict()
    net = net.double().eval()
    with torch.no_grad():
        net(torch.rand(1, 3, 16, 16, dtype=torch.float64))
    assert net.mean.dtype == torch.float64
