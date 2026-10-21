"""T2 A3: Degrader (two-stage OTF chain) and PairPool, with fixed kernels and seeds."""

import random
from itertools import starmap
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from neosr.config import ConfigError
from neosr.data import otf_degradation as od
from neosr.data.degradations import (
    random_add_gaussian_noise_pt,
    random_add_poisson_noise_pt,
)
from neosr.data.otf_degradation import Degrader, PairPool
from neosr.data.transforms import paired_random_crop
from neosr.utils.diffjpeg import DiffJPEG, filter2D
from neosr.utils.rng import rng, seed_rng

CFG = {
    "resize_prob": [0.3, 0.4, 0.3], "resize_range": [0.5, 1.5], "gaussian_noise_prob": 0.5,
    "noise_range": [0, 2], "poisson_scale_range": [0.05, 0.25], "gray_noise_prob": 0.3,
    "jpeg_range": [40, 95], "second_blur_prob": 0.5, "resize_prob2": [0.3, 0.4, 0.3],
    "resize_range2": [0.3, 1.5], "gaussian_noise_prob2": 0.5, "noise_range2": [0, 2],
    "poisson_scale_range2": [0.05, 0.1], "gray_noise_prob2": 0.3, "jpeg_range2": [35, 95],
}  # fmt: skip
S = 4
DEV = "cuda"  # DiffJPEG builds its parameters on CUDA; without JPEG the chain also runs on CPU
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="DiffJPEG needs CUDA")
NO_JPEG = {k: v for k, v in CFG.items() if not k.startswith("jpeg_range")} | {
    "jpeg_compress": False
}


def _kernel(b, k, seed):  # on CPU, as the loader delivers them
    w = torch.rand(b, k, k, generator=torch.Generator().manual_seed(seed))
    return w / w.sum((1, 2), keepdim=True)


def _batch(b=2, h=96, seed=0, dev=DEV):
    gt = torch.rand(b, 3, h, h, generator=torch.Generator().manual_seed(seed)).to(dev)
    return (
        gt,
        _kernel(b, 7, seed + 1),
        _kernel(b, 9, seed + 2),
        _kernel(b, 21, seed + 3),
    )


def _seed(n):
    random.seed(n)
    seed_rng(n)
    torch.manual_seed(n)


def _reference(cfg, jpeger, gt, k1, k2, sinc, r=rng):
    """models/otf.py feed_data degradation at c48252d, all keys read from cfg (T3 D9 fix);
    `jpeg_compress` honoured in both final orders (the fix under test)."""
    k1, k2, sinc = k1.to(gt.device), k2.to(gt.device), sinc.to(gt.device)
    jpeg = cfg.get("jpeg_compress", True)
    ori_h, ori_w = gt.size()[2:4]
    out = filter2D(gt, k1)
    updown_type = random.choices(["up", "down", "keep"], cfg["resize_prob"])[0]
    if updown_type == "up":
        scale = r.uniform(1, cfg["resize_range"][1])
    elif updown_type == "down":
        scale = r.uniform(cfg["resize_range"][0], 1)
    else:
        scale = 1
    mode = random.choice(["area", "bilinear", "bicubic"])
    out = F.interpolate(out, scale_factor=scale, mode=mode)
    if r.uniform() < cfg["gaussian_noise_prob"]:
        out = random_add_gaussian_noise_pt(out, sigma_range=cfg["noise_range"], clip=True,
                                           rounds=False, gray_prob=cfg["gray_noise_prob"])  # fmt: skip
    else:
        out = random_add_poisson_noise_pt(out, scale_range=cfg["poisson_scale_range"],
                                          gray_prob=cfg["gray_noise_prob"], clip=True, rounds=False)  # fmt: skip
    if jpeg:
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*cfg["jpeg_range"])
        out = jpeger(torch.clamp(out, 0, 1), quality=jpeg_p)
    if r.uniform() < cfg["second_blur_prob"]:
        out = filter2D(out, k2)
    updown_type = random.choices(["up", "down", "keep"], cfg["resize_prob2"])[0]
    if updown_type == "up":
        scale = r.uniform(1, cfg["resize_range2"][1])
    elif updown_type == "down":
        scale = r.uniform(cfg["resize_range2"][0], 1)
    else:
        scale = 1
    mode = random.choice(["area", "bilinear", "bicubic"])
    out = F.interpolate(
        out, size=(int(ori_h / S * scale), int(ori_w / S * scale)), mode=mode
    )
    if r.uniform() < cfg["gaussian_noise_prob2"]:
        out = random_add_gaussian_noise_pt(out, sigma_range=cfg["noise_range2"], clip=True,
                                           rounds=False, gray_prob=cfg["gray_noise_prob2"])  # fmt: skip
    else:
        out = random_add_poisson_noise_pt(out, scale_range=cfg["poisson_scale_range2"],
                                          gray_prob=cfg["gray_noise_prob2"], clip=True, rounds=False)  # fmt: skip
    if r.uniform() < 0.5:
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(out, size=(ori_h // S, ori_w // S), mode=mode)
        out = filter2D(out, sinc)
        if jpeg:
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*cfg["jpeg_range2"])
            out = jpeger(torch.clamp(out, 0, 1), quality=jpeg_p)
    else:
        if jpeg:
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*cfg["jpeg_range2"])
            out = jpeger(torch.clamp(out, 0, 1), quality=jpeg_p)
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(out, size=(ori_h // S, ori_w // S), mode=mode)
        out = filter2D(out, sinc)
    return torch.clamp((out * 255.0).round(), 0, 255) / 255.0


@cuda
def test_shapes_after_crop():
    _seed(0)
    d, p = Degrader(CFG, scale=S, device=DEV), 16
    gt, *ks = _batch(b=3)
    lq = d(gt, *ks)
    assert lq.shape == (3, 3, 96 // S, 96 // S)
    gt, lq = paired_random_crop([gt], lq, p, S)
    assert lq.shape == (3, 3, p, p)
    assert gt.shape == (3, 3, p * S, p * S)
    assert lq.min() >= 0
    assert lq.max() <= 1


@cuda
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_equals_reference_on_three_batches(seed):
    d = Degrader(CFG, scale=S, device=DEV)
    batches = [_batch(seed=10 * seed + i) for i in range(3)]
    _seed(seed)
    got = list(starmap(d, batches))
    _seed(seed)  # independent JPEG with the historical settings, not the one under test
    jpeger = DiffJPEG(differentiable=False).to(DEV)
    ref = [_reference(CFG, jpeger, *b) for b in batches]
    for g, r in zip(got, ref, strict=True):
        assert torch.equal(g, r)


class _FixedRng(SimpleNamespace):
    """uniform() -> v (branch draws), uniform(lo, hi) -> midpoint."""

    def uniform(self, lo=None, hi=None):
        return self.v if lo is None else (lo + hi) / 2


@pytest.mark.parametrize("v", [0.0, 0.9])  # 0.0: resize+sinc then JPEG; 0.9: JPEG first
@pytest.mark.parametrize(
    ("jpeg", "dev"), [pytest.param(True, DEV, marks=cuda), (False, "cpu")]
)
def test_jpeg_compress_honoured_in_both_orders(monkeypatch, v, jpeg, dev):
    cfg = CFG if jpeg else NO_JPEG  # the JPEG ranges are then not required
    r = _FixedRng(v=v)
    monkeypatch.setattr(od, "rng", r)
    d = Degrader(cfg, scale=S, device=dev)
    calls, ref_jpeger = [], None
    if jpeg:
        inner = d.jpeger
        d.jpeger = lambda x, quality: calls.append(1) or inner(x, quality=quality)
        ref_jpeger = DiffJPEG(differentiable=False).to(dev)
    else:
        assert d.jpeger is None
    b = _batch(dev=dev)
    _seed(5)
    got = d(*b)
    assert len(calls) == (2 if jpeg else 0)
    _seed(5)
    ref = _reference(cfg, ref_jpeger, *b, r=r)
    assert torch.equal(got, ref)


def test_outputs_feed_autograd():
    """no_grad, not inference_mode (T2 8.5): degraded and pooled batches stay usable in backward."""
    _seed(0)
    d, pool = Degrader(NO_JPEG, scale=S, device="cpu"), PairPool(4, 2, device="cpu")
    conv = nn.Conv2d(3, 3, 3)
    for i in range(3):  # 2 filling calls, then 1 swap
        gt, *ks = _batch(seed=i, dev="cpu")
        lq = d(gt.requires_grad_(), *ks)
        assert not lq.requires_grad
        assert not lq.is_inference()
        lq, gt = pool(lq, gt.detach())
        assert not lq.is_inference()
        assert not gt.is_inference()
        conv.zero_grad()
        (conv(lq).mean() + conv(gt).mean()).backward()
        assert conv.weight.grad is not None


@pytest.mark.parametrize(
    "key", ["resize_prob", "jpeg_range2", "second_blur_prob", "gray_noise_prob2"]
)
def test_missing_key_raises(key):
    with pytest.raises(ConfigError, match=key):
        Degrader({k: v for k, v in CFG.items() if k != key}, scale=S, device=DEV)
    with pytest.raises(ConfigError, match=key):
        Degrader({**CFG, key: None}, scale=S, device=DEV)


def test_bad_range_raises():
    with pytest.raises(ConfigError, match="resize_range2"):
        Degrader({**CFG, "resize_range2": [0.3]}, scale=S, device=DEV)


def _ids(start, b=2):
    """Batch whose sample i carries the value start+i in both lq (b,3,2,2) and gt (b,3,8,8)."""
    v = torch.arange(start, start + b, dtype=torch.float32).view(b, 1, 1, 1)
    return v.expand(b, 3, 2, 2).clone(), v.expand(b, 3, 8, 8).clone()


def test_pair_pool():
    torch.manual_seed(0)
    pool = PairPool(7, 2, device="cpu")
    assert pool.size == 6
    for i in range(3):  # filling: pass-through, stored as copies
        lq, gt = _ids(2 * i)
        out_lq, out_gt = pool(lq, gt)
        assert out_lq is lq
        assert out_gt is gt
        lq.add_(100)
    assert sorted(pool.lq[:, 0, 0, 0].tolist()) == list(range(6))
    seen = []
    for i in range(3, 20):
        lq, gt = _ids(2 * i)
        out_lq, out_gt = pool(lq, gt)
        assert out_lq.shape == lq.shape
        assert out_gt.shape == gt.shape
        a, g = out_lq[:, 0, 0, 0], out_gt[:, 0, 0, 0]
        assert torch.equal(a, g)  # pairs stay matched
        assert torch.equal(out_lq, a.view(-1, 1, 1, 1).expand_as(out_lq))
        assert (
            out_lq.untyped_storage().data_ptr() != pool.lq.untyped_storage().data_ptr()
        )
        seen += a.tolist()
        held = pool.lq[:, 0, 0, 0].tolist()
        assert torch.equal(pool.lq[:, 0, 0, 0], pool.gt[:, 0, 0, 0])
        assert {2 * i, 2 * i + 1} <= set(held)
        out_lq.add_(1000)  # returned tensors are not views of the pool
        assert max(pool.lq.flatten().tolist()) < 1000
    # every pair id is either held or returned, exactly once
    assert sorted(seen + held) == list(range(40))


def test_pair_pool_validation():
    with pytest.raises(ConfigError, match="queue_size"):
        PairPool(1, 2, device="cpu")
    pool = PairPool(6, 2, device="cpu")
    pool(*_ids(0))
    lq, _ = _ids(0)
    for bad in (_ids(0, b=4), _ids(0, b=3), _ids(0, b=0), (lq, _ids(0, b=3)[1])):
        with pytest.raises(ValueError, match="batch_size 2"):
            pool(*bad)
    assert pool.ptr == 2  # rejected batches leave the pool untouched
