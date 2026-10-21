import copy
import math
import warnings
from contextlib import nullcontext

import pytest
import torch
from torch import nn
from torch.nn.utils import get_total_norm

from neosr.models.phase import Ema, LossMeter, Phase, no_sync

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
PRECISIONS = {
    "fp32": (torch.float32, False),
    "bf16": (torch.bfloat16, True),
    "fp16": (torch.float16, True),
}
PREC = ["fp32", pytest.param("bf16", marks=CUDA), pytest.param("fp16", marks=CUDA)]
# extra fp32 gradient on the first conv weight: NaN, inf, or finite grads whose total norm overflows
BAD = {"nan": math.nan, "inf": math.inf, "overflow": 1e30}


class Ctx:
    """Duck-typed losses.base.Context: inputs plus .logs/.log()."""

    def __init__(self, lq, gt, factor=None) -> None:
        self.lq, self.gt, self.factor, self.logs = lq, gt, factor, {}

    def log(self, key, value) -> None:
        self.logs[key] = value.detach()


class Term:
    log_key = "l_g_pix"

    def __init__(self, net) -> None:
        self.net = net

    def compute(self, ctx):
        value = (self.net(ctx.lq).float() - ctx.gt).abs().mean()
        if ctx.factor is not None:
            value = value + self.net[0].weight.sum() * ctx.factor
        ctx.log(self.log_key, value)
        return value


def _data():
    g = torch.Generator(DEVICE).manual_seed(0)
    return (
        torch.rand(2, 3, 16, 16, generator=g, device=DEVICE),
        torch.rand(2, 3, 16, 16, generator=g, device=DEVICE),
    )


def _setup(prec="fp32", *, opt=None, **kw):
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.Conv2d(8, 3, 1)
    ).to(DEVICE)
    opt = (opt or (lambda ps: torch.optim.AdamW(ps, lr=1e-2, foreach=True)))(
        net.parameters()
    )
    dtype, amp = PRECISIONS[prec]
    kw = {"accumulate": 1, "grad_clip": 1.0} | kw
    phase = Phase(
        "g",
        list(net.parameters()),
        opt,
        [Term(net)],
        amp_dtype=dtype,
        use_amp=amp,
        **kw,
    )
    return net, opt, phase, Ema(net, 0.9)


def _update(phase, ema, factor=None) -> bool:
    phase.backward(Ctx(*_data(), factor))
    ok = phase.step()
    if ok:
        ema.update()
    return ok


def _params(net):
    return [p.detach().clone() for p in net.parameters()]


def _snapshot(net, ema, opt):
    return (
        _params(net),
        copy.deepcopy(ema.module.state_dict()),
        copy.deepcopy(opt.state_dict()),
    )


def _bytes(t):
    return t.detach().contiguous().reshape(-1).view(torch.uint8)


def _same(a, b) -> bool:
    if isinstance(a, torch.Tensor):  # bitwise: dtype, shape, signed zeros, NaN payloads
        return (a.dtype, a.shape) == (b.dtype, b.shape) and torch.equal(
            _bytes(a), _bytes(b)
        )
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple):
        return len(a) == len(b) and all(map(_same, a, b))
    return a == b


@pytest.mark.parametrize("prec", PREC)
def test_first_good_update_copies_live_into_ema(prec):
    net, _, phase, ema = _setup(prec)
    assert _update(phase, ema)
    assert _same(ema.module.state_dict(), net.state_dict())
    assert _update(phase, ema)  # later updates lerp, so EMA now lags the live net
    assert not _same(list(ema.module.parameters()), list(net.parameters()))


@pytest.mark.parametrize("case", ["clean", *BAD])
@pytest.mark.parametrize("prec", PREC)
def test_skip_leaves_state_untouched(prec, case):
    net, opt, phase, ema = _setup(prec)
    for _ in range(2):  # EMA initialised and lerped, optimizer state populated
        assert _update(phase, ema)
    before = _snapshot(net, ema, opt)
    norm = phase.grad_norm

    ok = _update(phase, ema, BAD.get(case))

    assert ok == (case == "clean")
    assert all(p.grad is None for p in net.parameters())
    if phase.scaler.is_enabled():  # fp16: only an inf/NaN window halves the scale
        assert phase.scaler.get_scale() == (16.0 if case in ("nan", "inf") else 32.0)
    if ok:
        assert not _same(_params(net), before[0])
        return
    # parameters only: BN running stats are forward-mutated (T1 §8 risk 3)
    assert _same(_snapshot(net, ema, opt), before)
    assert phase.grad_norm is norm
    assert (phase.skipped, phase.streak) == (1, 1)
    assert _update(phase, ema)
    assert (phase.skipped, phase.streak) == (1, 0)


class Clock:
    """Stand-in scheduler counting step() calls, so optimizer state stays comparable."""

    n = 0

    def step(self) -> None:
        self.n += 1


@pytest.mark.parametrize("prec", PREC)
def test_streak_raises_at_max_skip_streak(prec):
    clock = Clock()
    net, opt, phase, ema = _setup(prec, max_skip_streak=3, schedulers=[clock])
    # the scheduler advances once per attempt, clean and skipped alike (D20)
    for i, factor in enumerate([None, None, math.nan, math.nan], 1):
        assert _update(phase, ema, factor) == (factor is None)
        assert clock.n == i
    before, scale = _snapshot(net, ema, opt), phase.scaler.get_scale()
    with pytest.raises(FloatingPointError, match="3 consecutive"):
        _update(phase, ema, math.nan)
    # the terminal attempt still cleans up before raising
    assert clock.n == 5
    assert all(p.grad is None for p in net.parameters())
    assert _same(_snapshot(net, ema, opt), before)
    if phase.scaler.is_enabled():
        assert phase.scaler.get_scale() == scale / 2


@CUDA
@pytest.mark.parametrize("prec", ["fp32", "bf16", "fp16"])
def test_host_syncs_per_update(prec):
    _, _, phase, ema = _setup(prec)
    for _ in range(2):
        _update(phase, ema)
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            assert _update(phase, ema)
        finally:
            torch.cuda.set_sync_debug_mode(0)
    syncs = [w for w in caught if "called a synchronizing" in str(w.message)]
    assert len(syncs) == (2 if prec == "fp16" else 1)


@pytest.mark.parametrize("grad_clip", [0.0, 1.0])
def test_grad_clip(grad_clip):
    net, opt, phase, _ = _setup(
        opt=lambda ps: torch.optim.SGD(ps, lr=0.0), grad_clip=grad_clip
    )
    seen, step = [], opt.step

    def record(*args, **kwargs):  # grads as the optimizer sees them
        seen.append([p.grad.clone() for p in net.parameters()])
        return step(*args, **kwargs)

    opt.step = record
    phase.backward(Ctx(*_data(), 10.0))  # norm well above 1
    grads = [p.grad.clone() for p in net.parameters()]
    norm = get_total_norm(grads)
    assert phase.step()
    assert torch.equal(phase.grad_norm, norm)  # logged norm is measured before clipping
    if grad_clip:
        assert get_total_norm(seen[0]).item() == pytest.approx(1.0, rel=1e-5)
    else:
        assert _same(seen[0], grads)


def test_accumulate_scales_once():
    net, _, phase, _ = _setup(opt=lambda ps: torch.optim.SGD(ps, lr=0.0), accumulate=2)
    ctx = Ctx(*_data())
    phase.backward(ctx)
    once = [p.grad.clone() for p in net.parameters()]
    phase.backward(Ctx(*_data()))
    # the logged objective is undivided
    assert torch.equal(ctx.logs["l_g_total"], ctx.logs["l_g_pix"])
    assert all(
        torch.allclose(p.grad, 2 * g)
        for p, g in zip(net.parameters(), once, strict=True)
    )
    ref = Term(net).compute(Ctx(*_data()))
    full = torch.autograd.grad(ref, list(net.parameters()))
    assert all(
        torch.allclose(p.grad, g) for p, g in zip(net.parameters(), full, strict=True)
    )


def test_no_sync():
    cm = nullcontext()

    class Ddp(nn.Module):
        def no_sync(self):
            return cm

    assert no_sync(Ddp(), boundary=False) is cm
    assert no_sync(Ddp(), boundary=True) is not cm
    assert isinstance(no_sync(nn.Linear(1, 1), boundary=False), nullcontext)


def test_ema_lerp_and_buffers():
    net = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
    ema = Ema(net, 0.75)
    assert not ema.module.training
    assert not any(p.requires_grad for p in ema.module.parameters())
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    ema.update()  # first update copies
    assert _same(ema.module.state_dict(), net.state_dict())
    before = _params(ema.module)
    with torch.no_grad():
        for p in net.parameters():
            p.mul_(2.0)
        net(torch.randn(8, 4))  # moves BN running stats
    ema.update()
    for e, b, live in zip(
        ema.module.parameters(), before, net.parameters(), strict=True
    ):
        assert torch.allclose(e, b + 0.25 * (live - b))
    assert _same(list(ema.module.buffers()), list(net.buffers()))


def test_loss_meter():
    t = torch.tensor
    m, first = LossMeter(), t(1.0)
    m.add({"a": first, "b": t(10.0)})
    m.add({"a": t(3.0)})
    m.add({"a": t(5.0)})
    assert m.read(reset=False) == {"a": 3.0, "b": 10.0}
    assert m.read(reset=False) == {"a": 3.0, "b": 10.0}  # sums kept
    assert first.item() == 1.0  # inputs are not aliased
    m.add({"b": t(20.0)})
    assert m.read() == {"a": 3.0, "b": 15.0}
    assert m.read() == {}
    m.add({"a": t(1.0), "c": t(math.nan)})
    with pytest.raises(FloatingPointError, match="'c'"):
        m.read()
