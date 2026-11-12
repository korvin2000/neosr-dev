"""scripts/compile_parity.py: the comparison and the verdict on synthetic one-update results, and
the driver on results already on disk (no child runs). CPU."""

import json

import pytest
import torch

from scripts.compile_parity import main, param_diffs, verdict

START = torch.linspace(-1, 1, 1000)
STEP = 1e-4 * torch.linspace(1, 2, 1000)
LOGS = {"l_g_pix": 0.03, "l_d_real": 0.66, "grad_norm_g": 0.7, "r1": 0.05}


def _res(noise=0.0, seed=0, logs=None, start=START) -> dict:
    """One update: start -> start + STEP + noise * |STEP| * N(0, 1), for G and D alike."""
    g = torch.Generator().manual_seed(seed)
    dp = STEP + noise * STEP * torch.randn(1000, generator=g)
    nets = ("G", "D")
    return {
        "logs": LOGS | (logs or {}),
        "before": {n: {"w": start.clone()} for n in nets},
        "after": {n: {"w": START + dp} for n in nets},
    }


def _set(**over) -> dict:
    """eager, the eager2 floor (exact), the bf16 scale (rho ~ 0.1, losses 1e-3)."""
    return {
        "eager": _res(),
        "eager2": _res(),
        "eager_fp32": _res(0.1, seed=1, logs={"l_g_pix": 0.03 * (1 + 1e-3)}),
        "compile": _res(0.12, seed=2),
        "compile_fp32": _res(0.05, seed=3),
    } | over


def test_param_diffs():
    d = param_diffs(_res(), _res(0.1, seed=1))["G"]
    assert d["same_start"]
    assert d["upd"] == pytest.approx(STEP.max().item(), rel=1e-3)
    assert d["rho"] == pytest.approx(0.1, rel=0.15)
    assert not param_diffs(_res(), _res())["D"]["rho"]  # exact
    assert not param_diffs(_res(), _res(start=START + 1e-7))["G"]["same_start"]


def test_verdict_pass_and_bounds():
    ok, why = verdict(_set(), tol=2.0)
    assert ok, why
    assert "losses 1.00e-03" in why[0]


def test_verdict_fails_on_params_losses_start_and_nan():
    ok, why = verdict(_set(compile=_res(0.5, seed=2)), tol=2.0)
    assert not ok
    assert any(w.startswith("compile: G rho") for w in why)
    # a loss beyond tol x the bf16 loss scale; ungated keys (grad norms, r1) never fail
    noisy = {"l_d_real": 0.66 * 1.01, "grad_norm_g": 1.0, "r1": 0.5}
    ok, why = verdict(_set(compile_fp32=_res(0.05, seed=3, logs=noisy)), tol=2.0)
    assert why == ["compile_fp32: l_d_real rel 1.00e-02 > 2 x 1.00e-03"]
    ok, why = verdict(_set(eager2=_res(start=START + 1e-7)), tol=2.0)
    assert why == [
        "eager2: G restored state differs from eager",
        "eager2: D restored state differs from eager",
    ]
    nan = _res(0.12, seed=2)
    nan["after"]["D"]["w"][0] = float("nan")
    assert not verdict(_set(compile=nan), tol=2.0)[0]


def test_verdict_floor_binds_and_missing_modes():
    # an eager rerun noisier than bf16 raises the bound: the comparison cannot resolve below it
    ok, _ = verdict(_set(compile=_res(0.5, seed=2), eager2=_res(0.4, seed=4)), tol=2.0)
    assert ok
    res = _set()
    del res["eager_fp32"]
    assert verdict(res, tol=2.0)[0] is None


def test_driver_reads_results_and_guards_settings(tmp_path, capsys):
    out = tmp_path / "out"
    for mode, r in _set().items():
        (out / mode).mkdir(parents=True)
        torch.save(r, out / mode / "result.pt")
    argv = ["cfg.toml", "8000.state", str(out)]
    assert main(argv) == 0
    assert "== PASS" in capsys.readouterr().out
    assert json.loads((out / "args.json").read_text())["fallback_random"] is False
    assert main([*argv, "--fallback-random"]) == 2  # another setting: no mixing
    torch.save(_res(0.5, seed=2), out / "compile" / "result.pt")
    assert main(argv) == 1
    assert "== FAIL" in capsys.readouterr().out
