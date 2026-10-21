import errno
import logging
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from neosr.config import ConfigError
from neosr.models import checkpoint as ck
from neosr.utils import rng as neosr_rng

LOG = logging.getLogger("test_checkpoint")
ESRGAN_G = Path(
    "/home/korvin/apps/neosr-dev/experiments/train_esrgan/models/net_g_2000.pth"
)


def _sd(**kw):
    return OrderedDict((k, torch.full((2,), float(v))) for k, v in kw.items())


# strip_prefixes


def test_strip_prefixes_nested_and_n_averaged():
    out = ck.strip_prefixes({
        "_orig_mod.module.x": 1,
        "module.y": 2,
        "n_averaged": 3,
        "module.n_averaged": 4,
    })
    assert out == OrderedDict(x=1, y=2)


def test_strip_prefixes_collision_raises():
    with pytest.raises(KeyError, match="collides"):
        ck.strip_prefixes({"module.x": 1, "x": 2})


def test_cpu_copy_detached_contiguous_independent():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    src = (
        torch.arange(24.0, device=dev).reshape(2, 3, 4).transpose(1, 2).requires_grad_()
    )
    assert not src.is_contiguous()
    out = ck.cpu_copy({"w": src})["w"]
    assert out.device.type == "cpu"
    assert out.is_contiguous()
    assert not out.requires_grad
    assert out.grad_fn is None
    assert torch.equal(out, src.detach().cpu())
    with torch.no_grad():
        src.add_(1)
    assert torch.equal(out + 1, src.detach().cpu())  # own storage


# read_weights


def test_read_weights_precedence(tmp_path):
    p = tmp_path / "w.pth"
    torch.save({"params": _sd(a=1), "params-ema": _sd(a=2), "params_ema": _sd(a=3)}, p)
    assert ck.read_weights(p)["a"][0] == 3
    torch.save({"params": _sd(a=1), "params-ema": _sd(a=2)}, p)
    assert ck.read_weights(p)["a"][0] == 2
    torch.save({"params": {"module.a": torch.ones(1)}}, p)
    assert list(ck.read_weights(p)) == ["a"]


def test_read_weights_explicit_key(tmp_path):
    p = tmp_path / "w.pth"
    torch.save({"params": _sd(a=1), "params_ema": _sd(a=3)}, p)
    assert ck.read_weights(p, "params")["a"][0] == 1
    with pytest.raises(ConfigError, match="param_key 'nope' not found"):
        ck.read_weights(p, "nope")


def test_read_weights_bare_dict(tmp_path):
    p = tmp_path / "w.pth"
    torch.save({"_orig_mod.a": torch.ones(1), "n_averaged": torch.tensor(5)}, p)
    assert list(ck.read_weights(p)) == ["a"]


def test_read_weights_rejects_state_and_unknown(tmp_path):
    p = tmp_path / "1.state"
    torch.save({"format": ck.STATE_FORMAT, "nets": {"net_g": _sd(a=1)}}, p)
    with pytest.raises(ConfigError, match="training state"):
        ck.read_weights(p)
    torch.save({"iter": 3, "optimizers": []}, p)
    with pytest.raises(ConfigError, match="no parameter mapping"):
        ck.read_weights(p)


@pytest.mark.skipif(not ESRGAN_G.is_file(), reason="esrgan export absent")
def test_read_weights_esrgan_export():
    sd = ck.read_weights(ESRGAN_G)
    assert sd
    assert all(isinstance(v, torch.Tensor) for v in sd.values())
    assert not any(k.startswith(ck._PREFIXES) for k in sd)


# load_weights


def test_load_weights_relaxed(caplog):
    net = nn.Sequential(nn.Conv2d(3, 4, 3), nn.Conv2d(4, 3, 3))
    sd = {
        "0.weight": torch.zeros(4, 3, 3, 3),
        "1.weight": torch.zeros(5, 4, 3, 3),
        "extra": torch.ones(1),
    }
    with caplog.at_level(logging.WARNING, logger=LOG.name):
        ck.load_weights(net, sd, strict=False, log=LOG)
    assert torch.equal(net[0].weight, torch.zeros(4, 3, 3, 3))
    assert "1.weight" in sd  # caller's dict untouched
    text = caplog.text
    assert "shape mismatch, skipped 1.weight" in text
    assert "missing: 1.weight" in text
    assert "unexpected: extra" in text


def test_load_weights_relaxed_zero_matches_raises():
    net = nn.Conv2d(3, 3, 1)
    with pytest.raises(ConfigError, match="matches no tensor"):
        ck.load_weights(
            net,
            {"weight": torch.zeros(1), "other": torch.zeros(1)},
            strict=False,
            log=LOG,
        )


def test_load_weights_strict_raises():
    with pytest.raises(RuntimeError):
        ck.load_weights(
            nn.Conv2d(3, 3, 1),
            {"weight": torch.zeros(3, 3, 1, 1)},
            strict=True,
            log=LOG,
        )


# atomic_save


def _no_tmp(d: Path) -> bool:
    return not list(d.glob("*.tmp"))


@pytest.fixture
def replace_fails(monkeypatch):
    """Make os.replace raise `exc` for the first `n` calls; returns the call log."""
    real, calls = ck.os.replace, []
    monkeypatch.setattr(ck.time, "sleep", lambda _: None)

    def arm(n, exc):
        def fake(src, dst):
            calls.append(src)
            if len(calls) <= n:
                raise exc
            real(src, dst)

        monkeypatch.setattr(ck.os, "replace", fake)
        return calls

    return arm


@pytest.fixture
def target(tmp_path, monkeypatch):
    """A destination holding a previous checkpoint; counts serializations."""
    p = tmp_path / "x.pth"
    ck.atomic_save({"a": 0}, p)
    old, real, saves = p.read_bytes(), ck.torch.save, []
    monkeypatch.setattr(
        ck.torch, "save", lambda *a, **k: saves.append(1) or real(*a, **k)
    )
    return p, old, saves


def test_atomic_save_retries_permission_error(tmp_path, replace_fails, target):
    p, _, saves = target
    calls = replace_fails(2, PermissionError("held"))
    ck.atomic_save({"a": 1}, p)
    assert len(calls) == 3
    assert len(saves) == 1  # serialized once, only the rename retried
    assert torch.load(p, weights_only=True) == {"a": 1}
    assert _no_tmp(tmp_path)


def test_atomic_save_gives_up_after_attempts(tmp_path, replace_fails, target):
    p, old, saves = target
    calls = replace_fails(3, PermissionError("held"))
    with pytest.raises(PermissionError):
        ck.atomic_save({"a": 1}, p)
    assert len(calls) == 3
    assert len(saves) == 1
    assert p.read_bytes() == old
    assert _no_tmp(tmp_path)


def test_atomic_save_no_retry_on_enospc(tmp_path, replace_fails, target):
    p, old, _ = target
    calls = replace_fails(1, OSError(errno.ENOSPC, "full"))
    with pytest.raises(OSError, match="full") as e:
        ck.atomic_save({"a": 1}, p)
    assert e.value.errno == errno.ENOSPC
    assert len(calls) == 1
    assert p.read_bytes() == old
    assert _no_tmp(tmp_path)


def test_atomic_save_error_in_save(tmp_path, monkeypatch, replace_fails, target):
    p, old, _ = target
    calls = replace_fails(0, None)

    def boom(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(ck.torch, "save", boom)
    with pytest.raises(RuntimeError, match="boom"):
        ck.atomic_save({"a": 1}, p)
    assert calls == []
    assert p.read_bytes() == old
    assert _no_tmp(tmp_path)


@pytest.mark.parametrize("attempts", [1, 0, -1])
def test_atomic_save_always_writes_once(tmp_path, replace_fails, attempts):
    calls = replace_fails(0, None)
    ck.atomic_save({"a": 1}, tmp_path / "x.pth", attempts=attempts)
    assert len(calls) == 1
    assert torch.load(tmp_path / "x.pth", weights_only=True) == {"a": 1}
    assert _no_tmp(tmp_path)


def test_atomic_save_overwrites(tmp_path):
    p = tmp_path / "x.pth"
    ck.atomic_save({"a": 1}, p)
    ck.atomic_save({"a": 2}, p)
    assert torch.load(p, weights_only=True) == {"a": 2}
    assert _no_tmp(tmp_path)


# latest_state / prune_states


def _states(d: Path, *stems):
    for s in stems:
        (d / f"{s}.state").write_bytes(b"")


def test_latest_state_filters(tmp_path):
    assert ck.latest_state(tmp_path) is None
    _states(tmp_path, 2, 10, "best", "x1")
    (tmp_path / ".100.state.abc.tmp").write_bytes(b"")
    (tmp_path / "200.state").mkdir()
    assert ck.latest_state(tmp_path) == tmp_path / "10.state"


def test_prune_states(tmp_path):
    _states(tmp_path, 1, 2, 3, 10, 20, "best")
    ck.prune_states(tmp_path, 0)
    assert len(list(tmp_path.glob("*.state"))) == 6
    ck.prune_states(tmp_path, 3)
    assert sorted(p.name for p in tmp_path.glob("*.state")) == [
        "10.state",
        "20.state",
        "3.state",
        "best.state",
    ]


# RNG


@pytest.fixture
def neosr_stream(monkeypatch):
    """The T3 neosr.utils.rng stream; a PCG64 stand-in until get_state/set_state land."""
    if hasattr(neosr_rng, "get_state"):
        return lambda: neosr_rng.rng.random(4)
    gen = np.random.default_rng(1)
    monkeypatch.setattr(
        neosr_rng, "get_state", lambda: gen.bit_generator.state, raising=False
    )
    monkeypatch.setattr(
        neosr_rng,
        "set_state",
        lambda s: setattr(gen.bit_generator, "state", s),
        raising=False,
    )
    return lambda: gen.random(4)


def _draws(neosr_draw, loader_gen):
    return (
        [random.random() for _ in range(4)],
        np.random.rand(4).tolist(),
        np.random.randn(3).tolist(),  # exercises the cached gaussian
        neosr_draw().tolist(),
        torch.rand(4).tolist(),
        torch.rand(4, generator=loader_gen).tolist(),
    )


def test_rng_round_trip(tmp_path, neosr_stream):
    loader_gen = torch.Generator().manual_seed(7)
    np.random.seed(11)
    np.random.randn()  # odd draw count leaves a cached gaussian behind
    assert np.random.get_state()[3] == 1
    ck.atomic_save(ck.rng_state(None, loader_gen), tmp_path / "r.state")
    expected = _draws(neosr_stream, loader_gen)
    _draws(neosr_stream, loader_gen)  # advance every stream further
    s = torch.load(tmp_path / "r.state", map_location="cpu", weights_only=True)
    ck.set_rng_state(s, None, loader_gen)
    assert _draws(neosr_stream, loader_gen) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_rng_round_trip_cuda(tmp_path):
    dev = torch.device("cuda", torch.cuda.current_device())
    ck.atomic_save(ck.rng_state(dev), tmp_path / "r.state")
    expected = torch.rand(4, device=dev)
    s = torch.load(tmp_path / "r.state", map_location="cpu", weights_only=True)
    ck.set_rng_state(s, dev)
    assert torch.equal(torch.rand(4, device=dev), expected)


def test_preserved_rng(neosr_stream):
    loader_gen = torch.Generator().manual_seed(3)
    before = ck.rng_state(None, loader_gen)
    with ck.preserved_rng():
        _draws(neosr_stream, torch.Generator())
    expected = _draws(neosr_stream, loader_gen)
    ck.set_rng_state(before, None, loader_gen)
    assert _draws(neosr_stream, loader_gen) == expected
