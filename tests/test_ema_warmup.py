"""EMA decay warmup (DECISIONS #9): ema_pytorch-style ramp, its validation and exact resume."""

import io

import pytest
import torch
from torch import nn

from neosr.config import ConfigError, validate_train
from neosr.models.phase import Ema
from tests.test_resume import assert_same, continuation, make_opt
from tests.test_resume import deterministic as deterministic  # autouse fixture


def net() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4))


def drift(n: nn.Module, k: int) -> None:
    with torch.no_grad():
        for t in (*n.parameters(), *n.buffers()):
            if t.is_floating_point():
                t.add_(k)


def roundtrip(sd: dict) -> dict:
    buf = io.BytesIO()
    torch.save(sd, buf)
    buf.seek(0)
    return torch.load(buf, weights_only=True)


@pytest.mark.parametrize("power", [None, 0.75])
@torch.no_grad()
def test_kth_lerp_uses_the_ramped_decay(power):
    live, decay = net(), 0.6  # the ramp reaches it at k = 3
    ema = Ema(live, decay, power)
    ema.update()  # copy
    ref = [p.clone() for p in live.parameters()]
    for k in range(1, 8):
        drift(live, k)
        ema.update()
        d = decay if power is None else min(1 - (1 + k) ** -power, decay)
        ref = [
            r + (1 - d) * (p - r) for r, p in zip(ref, live.parameters(), strict=True)
        ]
        for r, e in zip(ref, ema.module.parameters(), strict=True):
            torch.testing.assert_close(e, r, rtol=1e-6, atol=1e-5)
        assert ema.current_decay() == d
    assert ema.current_decay() == decay
    assert ema.lerps == 7


def test_ramp_matches_ema_pytorch_and_clamps():
    ema = Ema(net(), 0.999, 0.75)
    for k, want in ((0, 0.0), (1, 1 - 2**-0.75), (9, 1 - 10**-0.75)):
        ema.lerps = k
        assert ema.current_decay() == pytest.approx(want)
    ema.lerps = 10**6  # 1 - 1e6**-0.75 > 0.999
    assert ema.current_decay() == 0.999


@torch.no_grad()
def test_state_carries_the_count_only_with_a_power():
    live = net()
    plain, ramp = Ema(live, 0.9), Ema(live, 0.9, 0.75)
    for e in (plain, ramp):
        e.update()
        drift(live, 1)
        e.update()
    assert Ema.KEY not in plain.state_dict()  # format unchanged without warmup
    sd = roundtrip(ramp.state_dict())
    assert int(sd[Ema.KEY]) == 1
    keys = set(ramp.module.state_dict())
    assert set(sd) - {Ema.KEY} == keys

    again = Ema(live, 0.9, 0.75)
    again.load_state_dict(sd)
    assert Ema.KEY in sd  # the caller's dict is left intact
    assert again.lerps == 1
    drift(live, 2)
    for e in (ramp, again):
        e.update()
    assert_same(again.module.state_dict(), ramp.module.state_dict())


@torch.no_grad()
def test_a_state_without_the_count_loads_warmed_up():
    old = Ema(net(), 0.9)
    old.update()
    ema = Ema(net(), 0.9, 0.75)
    ema.load_state_dict(roundtrip(old.state_dict()))
    assert ema.lerps is None
    assert ema.current_decay() == 0.9
    ema.update()  # stays warmed up, and so does its state
    assert Ema.KEY not in ema.state_dict()
    # a warmup run resumed without ema_power ignores the count
    plain = Ema(net(), 0.9)
    warm = Ema(net(), 0.9, 0.75)
    warm.update()
    plain.load_state_dict(warm.state_dict())
    assert plain.current_decay() == 0.9


def cfg(**train) -> dict:
    return {
        "scale": 2,
        "network_g": {"type": "compact"},
        "datasets": {"train": {"patch_size": 8, "batch_size": 2}},
        "train": {"optim_g": {"type": "adamw"}, "ema": 0.999, **train},
    }


@pytest.mark.parametrize("power", [0, -0.5, float("nan")])
def test_ema_power_must_be_positive(power):
    with pytest.raises(ConfigError, match="ema_power"):
        validate_train(cfg(ema_power=power))
    validate_train(cfg(ema_power=0.75))
    validate_train(cfg())


def test_resume_with_warmup_is_exact(tmp_path):
    def make(t):
        opt = make_opt(t, "a")
        opt["train"]["ema_power"] = 0.75
        return opt

    a, r, sa, sr = continuation(tmp_path, make)
    assert a.ema.power == r.ema.power == 0.75
    assert int(sa["ema"][Ema.KEY]) == 5  # 6 updates: one copy, five lerps
    assert_same(sa, sr)
