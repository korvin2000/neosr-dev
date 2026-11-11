"""scripts/ab on synthetic runs: analyze.py's cluster bootstrap, gate and input guards, and
queue.sh with a stub trainer. CPU."""

import json
import os
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.ab.analyze import (
    Data,
    bootstrap,
    diagnostics,
    main,
    paired,
    status,
    verdict,
)

IT, N_GROUPS, PER_GROUP = 8000, 20, 5
BASE = {"psnr": 30.0, "topiq": 0.65, "lpips": 0.17, "dists": 0.15}


def _ids() -> list[str]:
    return [f"g{g:02d}i{i}" for g in range(N_GROUPS) for i in range(PER_GROUP)]


def _data(tmp, exclude=(), n_groups=N_GROUPS) -> None:
    (tmp / "dev_gt").mkdir()
    for i in _ids():
        (tmp / "dev_gt" / f"{i}.png").touch()
    groups = [
        [f"{i}.png" for k, i in enumerate(_ids()) if k * n_groups // len(_ids()) == g]
        for g in range(n_groups)
    ]
    (tmp / "sources.json").write_text(json.dumps(groups))
    (tmp / "exclude.txt").write_text("\n".join(exclude))


def _arm(runs, name, shift=None, seed=0, drop=0, skip=(), its=(IT,), at=None) -> None:
    """Per-image rows at each update of `its`: a fixed per-image base (shared by every arm and
    update), small noise, plus `shift` (only at update `at`, if given)."""
    noise = np.random.default_rng(seed)
    (runs / name).mkdir(parents=True)
    rows = []
    for it in its:
        base = np.random.default_rng(99)
        for i in _ids()[drop:]:
            row = {"iter": it, "dataset": "dev", "id": i}
            for m, v in BASE.items():
                if m in skip:
                    continue
                scale = 0.5 if m == "psnr" else 0.01
                row[m] = (
                    v
                    + scale * base.normal()
                    + 1e-4 * noise.normal()
                    + (shift or {}).get(m, 0.0) * (at is None or it == at)
                )
            rows.append(row)
    lines = "".join(json.dumps(r) + "\n" for r in rows)
    (runs / name / "per_image.jsonl").write_text(lines)


def test_bootstrap_resamples_groups_not_images():
    """Differences shared inside each group: the CI is that of 20 group means, ~sqrt(5) wider than
    the CI of 100 independent images; a constant difference gives a zero-width CI."""
    g = np.repeat(np.arange(N_GROUPS), PER_GROUP)
    d = np.repeat(np.random.default_rng(0).normal(0.01, 0.02, N_GROUPS), PER_GROUP)
    mean, lo, hi = bootstrap(d, g)
    _, lo_i, hi_i = bootstrap(d, np.arange(len(d)))
    assert lo < mean < hi
    assert mean == pytest.approx(d.mean())
    assert (hi - lo) > 1.8 * (hi_i - lo_i)
    assert bootstrap(np.full(10, 0.3), np.arange(10))[1:] == pytest.approx((0.3, 0.3))


def test_status_and_verdict():
    assert status("lpips", -0.01, -0.012, -0.008, 0.002) == "better"
    assert status("lpips", -0.01, -0.02, 0.001, 0.002) == "same"  # CI includes 0
    assert status("topiq", 0.001, 0.0005, 0.0015, 0.0066) == "same"  # under the floor
    assert status("psnr", -0.2, -0.3, -0.1, 0.05) == "worse"
    assert verdict({"topiq": "better", "lpips": "better", "dists": "same"}) == "WIN"
    assert verdict({"topiq": "better", "lpips": "same", "dists": "same"}) == "TIE"
    assert verdict({"topiq": "better", "lpips": "better", "dists": "worse"}) == "LOSS"
    assert (
        verdict({"topiq": "same", "lpips": "same", "dists": "same", "psnr": "worse"})
        == "TIE"
    )
    both = {"topiq": "better", "lpips": "better", "dists": "same"}
    one = {"topiq": "same", "lpips": "better", "dists": "better"}
    assert verdict(both, ("topiq", "lpips")) == "WIN"  # two gated: both must be better
    assert verdict(one, ("topiq", "lpips")) == "TIE"  # dists reported only
    assert verdict(one, ("lpips",)) == "WIN"


def test_gate_end_to_end(tmp_path, capsys):
    data, runs = tmp_path / "data", tmp_path / "runs"
    data.mkdir()
    _data(data, exclude=["g00i0", "g00i1"])
    _arm(runs, "B0", seed=0)
    _arm(runs, "B0s2", seed=1)
    _arm(runs, "win", {"topiq": 0.02, "lpips": -0.01, "psnr": -0.02}, seed=2)
    _arm(runs, "loss", {"topiq": 0.02, "lpips": -0.01, "dists": 0.01}, seed=3)
    _arm(runs, "tie", seed=4)
    argv = [
        "--runs",
        str(runs),
        "--data",
        str(data),
        "--seed-ref",
        "B0s2",
        "--iter",
        str(IT),
    ]
    assert main([*argv, "B0", "win", "loss", "tie"]) == 0
    out = {
        line.split(":")[0]: line
        for line in capsys.readouterr().out.splitlines()
        if "|" in line
    }
    assert out["win"].startswith("win: WIN")
    assert out["loss"].startswith("loss: LOSS")
    assert out["tie"].startswith("tie: TIE")
    assert (
        "n=98, groups=20" in out["win"]
    )  # the excluded ids left out, groups from sources.json


def test_guards(tmp_path):
    data, runs = tmp_path / "data", tmp_path / "runs"
    data.mkdir()
    _data(data)
    _arm(runs, "B0")
    _arm(runs, "short", drop=1)  # one manifest image missing
    with pytest.raises(SystemExit, match="manifest 100"):
        main(["--runs", str(runs), "--data", str(data), "B0", "short"])
    a = {"x": {"psnr": 1.0}}
    with pytest.raises(SystemExit, match="different image sets"):
        paired(a, {"y": {"psnr": 1.0}}, Data({}, {}, set()))


def test_seed_spread_binds_the_floor(tmp_path, capsys):
    """A second seed farther from the reference than the arm's gain: the floor rises, WIN -> TIE."""
    data, runs = tmp_path / "data", tmp_path / "runs"
    data.mkdir()
    _data(data)
    _arm(runs, "B0", seed=0)
    _arm(runs, "B0s2", {"topiq": 0.03, "lpips": -0.015}, seed=1)
    _arm(runs, "win", {"topiq": 0.02, "lpips": -0.01}, seed=2)
    argv = ["--runs", str(runs), "--data", str(data), "--iter", str(IT)]
    main([*argv, "B0", "win"])
    assert "win: WIN" in capsys.readouterr().out
    main([*argv, "--seed-ref", "B0s2", "B0", "win"])
    out = capsys.readouterr().out
    assert "win: TIE" in out
    assert "topiq 0.0300" in out


def test_missing_inputs_exit(tmp_path):
    data, runs = tmp_path / "data", tmp_path / "runs"
    data.mkdir()
    _data(data)
    _arm(runs, "B0")
    _arm(runs, "nolpips", skip=("lpips",))
    argv = ["--runs", str(runs), "--data", str(data), "--iter", str(IT)]
    with pytest.raises(SystemExit, match="for the seed spread"):
        main([*argv, "--seed-ref", "absent", "B0", "B0"])
    with pytest.raises(SystemExit, match="gate incomplete, no lpips"):
        main([*argv, "B0", "nolpips"])
    (data / "sources.json").unlink()
    with pytest.raises(SystemExit, match="no sources file"):
        main([*argv, "B0", "B0"])
    assert main([*argv, "--independent", "B0", "B0"]) == 0
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"sources": "data/gone.json", "manifest": {}}))
    with pytest.raises(SystemExit, match=r"sources file .* is missing"):
        main(["--runs", str(runs), "--data", str(spec), "B0", "B0"])
    spec.write_text(json.dumps({"manifest": {}}))
    with pytest.raises(SystemExit, match="no manifest"):
        main(["--runs", str(runs), "--data", str(spec), "--independent", "B0", "B0"])


def test_one_source_group_exits(tmp_path):
    data, runs = tmp_path / "data", tmp_path / "runs"
    data.mkdir()
    _data(data, n_groups=1)
    _arm(runs, "B0")
    with pytest.raises(SystemExit, match="1 source group"):
        main(["--runs", str(runs), "--data", str(data), "--iter", str(IT), "B0", "B0"])


def test_nested_manifest_ids_follow_the_trainer(tmp_path):
    from scripts.ab.analyze import load_data  # ruff: ignore[import-outside-top-level]

    (tmp_path / "dev_gt" / "sub").mkdir(parents=True)
    (tmp_path / "dev_gt" / "sub" / "a.png").touch()
    (tmp_path / "dev_gt" / "b.png").touch()
    (tmp_path / "sources.json").write_text(json.dumps([["sub/a.png"], ["b.png"]]))
    data = load_data(tmp_path)
    assert data.manifest["dev"] == {str(Path("sub/a")), "b"}
    assert data.group[str(Path("sub/a"))] != data.group["b"]


QUEUE = Path(__file__).resolve().parents[1] / "scripts" / "ab" / "queue.sh"
STUB = """#!/bin/bash
# run_ab.py stand-in: $2 code, $3 toml, $4 out; arm b fails, others finish
out=$4
echo "$3" > "$out/ran"
[ "$(basename "$out")" = b ] && exit 3
echo '{}' > "$out/done.json"
"""


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None, reason="bash queue"
)
def test_queue_relative_paths_failing_rc_and_blank_lines(tmp_path):
    (tmp_path / "cfg").mkdir()
    for arm in ("a", "b", "c"):
        (tmp_path / "cfg" / f"{arm}.toml").touch()
    stub = tmp_path / "stub.sh"
    stub.write_text(STUB)
    stub.chmod(0o755)
    (tmp_path / "q.txt").write_text("a\n\n# later: c\nb\r\n")
    env = os.environ | {
        "CFG": "cfg",
        "RUNS": "runs",
        "QUEUE": "q.txt",
        "PY": str(stub),
        "DEADLINE": "23:59",
        "SINGLE_MIN": "0",
        "MARGIN_MIN": "0",
    }
    env.pop("LOCK", None)
    out = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [shutil.which("bash"), str(QUEUE)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout
    assert "end a rc=0 done=yes" in out
    assert "end b rc=3 done=NO" in out
    assert not (tmp_path / "runs" / "c").exists()
    assert (tmp_path / "runs" / "b" / "ran").exists()
    assert not (tmp_path / "q.txt").read_text()


WIN_ITS = (4000, 6000, 8000, 10000)


def _window(tmp_path, null_shift=None):
    """Three same-config runs (A0, B0, B0s2), validated at WIN_ITS; B0s2 optionally offset."""
    data, runs = tmp_path / "data", tmp_path / "runs"
    data.mkdir()
    _data(data)
    _arm(runs, "A0", seed=0, its=WIN_ITS)
    _arm(runs, "B0", seed=1, its=WIN_ITS)
    _arm(runs, "B0s2", null_shift, seed=2, its=WIN_ITS)
    argv = ["--runs", str(runs), "--data", str(data), "--window", "4000:8000"]
    return runs, [*argv, "--null", "A0/B0,B0/B0s2"]


def _verdicts(capsys) -> dict[str, str]:
    return {
        line.split(":")[0]: line
        for line in capsys.readouterr().out.splitlines()
        if "|" in line
    }


def test_window_gate(tmp_path, capsys):
    """D10: the window's updates only (10000 left out), a pooled reference, LOSS vs mixed LOSS,
    and TOPIQ judged without the 1 % term (+0.002 < 1 % of 0.65 is still better)."""
    runs, argv = _window(tmp_path)
    _arm(runs, "win", {"topiq": 0.002, "lpips": -0.01}, seed=3, its=WIN_ITS)
    _arm(runs, "loss", {"dists": 0.01}, seed=4, its=WIN_ITS)
    _arm(runs, "mixed", {"lpips": -0.01, "dists": 0.01}, seed=5, its=WIN_ITS)
    _arm(runs, "late", {"lpips": 0.05}, seed=6, its=(10000,))  # outside the window
    _arm(runs, "short", seed=7, its=(4000, 6000))  # misses 8000
    assert main([*argv, "A0+B0", "win", "loss", "mixed", "short"]) == 0
    out = _verdicts(capsys)
    assert out["win"].startswith("win: WIN")
    assert out["loss"].startswith("loss: LOSS |")
    assert out["mixed"].startswith("mixed: LOSS (mixed)")
    assert "short" not in out  # not evaluated: a window update is missing
    main([*argv, "A0", "late"])
    assert "late: not evaluated" in capsys.readouterr().out


def test_window_null_rms_binds_tau(tmp_path, capsys):
    """A same-config pair 0.004 apart in LPIPS: tau = 3 x RMS(~0, 0.004) = 0.0085 > 0.005."""
    runs, argv = _window(tmp_path, {"lpips": -0.004})
    _arm(runs, "gain", {"lpips": -0.005}, seed=3, its=WIN_ITS)
    main([*argv, "A0", "gain"])
    out = capsys.readouterr().out
    assert "lpips 0.0085" in out.split("## tau:")[1].splitlines()[0]
    assert "gain: TIE" in out


def test_report_only(tmp_path, capsys):
    """An arm trained on DISTS: its DISTS gain is reported, and one gated gain is a TIE."""
    runs, argv = _window(tmp_path)
    _arm(runs, "d", {"lpips": -0.01, "dists": -0.01}, seed=3, its=WIN_ITS)
    main([*argv, "A0", "d"])
    assert "d: WIN" in capsys.readouterr().out
    main([*argv, "--report-only", "dists", "A0", "d"])
    out = capsys.readouterr().out
    assert "d: TIE" in out
    assert "## gated: topiq, lpips; reported: psnr, dists" in out


def test_window_means_over_updates_and_pooled_runs(tmp_path, capsys):
    """+0.03 LPIPS at one of three updates is +0.01 over the window; a pooled reference whose
    second run is 0.004 lower sits 0.002 below the first run alone."""
    runs, argv = _window(tmp_path, {"lpips": -0.004})
    _arm(runs, "once", {"lpips": 0.03}, seed=3, its=WIN_ITS, at=6000)
    main([*argv, "A0+B0s2", "once"])
    line = _verdicts(capsys)["once"]
    assert "lpips +0.0120 " in line  # 0.03 / 3 + 0.004 / 2
    main([*argv, "A0", "once"])
    assert "lpips +0.0100 " in _verdicts(capsys)["once"]


def test_window_guards(tmp_path, capsys):
    runs, argv = _window(tmp_path)
    base = argv[:-2]  # without --null
    with pytest.raises(SystemExit):
        main([*base, "A0", "B0"])  # --window needs --null
    with pytest.raises(SystemExit):
        main([*argv, "--seed-ref", "B0s2", "A0", "B0"])
    with pytest.raises(SystemExit, match="FROM:TO"):
        main([*argv[:-4], "--window", "8000", "--null", "A0/B0", "A0", "B0"])
    with pytest.raises(SystemExit):
        main([*base, "--null", "A0", "A0", "B0"])
    assert "X/Y pairs" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="no dev validation in 20000:30000"):
        main([*argv[:-4], "--window", "20000:30000", "--null", "A0/B0", "A0", "B0"])
    with pytest.raises(SystemExit):
        main([
            *base,
            "--null",
            "B0/B0",
            "A0",
            "B0",
        ])  # a run is no noise pair with itself
    with pytest.raises(SystemExit):
        main([
            *argv,
            *(f"--report-only={m}" for m in ("topiq", "lpips", "dists")),
            "A0",
            "B0",
        ])
    # a reference run cut short (no 8000): the gate stops instead of shortening the window
    _arm(runs, "cut", seed=8, its=(4000, 6000))
    with pytest.raises(SystemExit, match="cut: no dev rows"):
        main([*base, "--null", "B0/B0s2", "cut", "B0"])


def test_diagnostics_speed_and_memory(tmp_path):
    """ms/update: the median over log windows, a validation pause and a resume's restarted clock
    left out; mem_gib: the first (start-up) and the steady window peak."""
    (tmp_path / "a").mkdir()
    t = [
        (100, 10.0),
        (200, 20.0),
        (300, 30.0),
        (400, 70.0),
        (500, 80.0),
        (600, 1.0),
        (700, 11.0),
    ]
    mem = [12.5, 4.7, 4.7, 4.7, 4.7, 12.4, 4.7]
    rows = [{"iter": i, "t": s, "mem_gib": m} for (i, s), m in zip(t, mem, strict=True)]
    (tmp_path / "a" / "train.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    d = diagnostics(tmp_path, "a")
    assert d["ms_update"] == pytest.approx(100.0)
    assert d["mem_gib"] == {"first": 12.5, "steady": 4.7, "max": 12.5}


def test_run_ab_records_the_window_peak(tmp_path, monkeypatch):
    """run_ab's hooks put the [ mem ] peak train.py reads just before get_current_log into that
    log row, once; every patched attribute is restored after the test."""
    import importlib  # ruff: ignore[import-outside-top-level]

    from scripts.ab.run_ab import Recorder, install_hooks  # ruff: ignore[import-outside-top-level]

    base = importlib.import_module("neosr.models.base")
    image = importlib.import_module("neosr.models.image")
    trainer = importlib.import_module("train")
    for obj, attr in (
        (base.base, "log_metrics"),
        (image.image, "validate"),
        (image.image, "_metric"),
        (image, "image_id"),
    ):
        monkeypatch.setattr(obj, attr, getattr(obj, attr))
    monkeypatch.setattr(image.image, "get_current_log", lambda _self: {"l_g_pix": 1.0})
    monkeypatch.setattr(trainer, "window_peak_memory", lambda _device: (4.7, 23.6))
    rec = Recorder(tmp_path)
    install_hooks(rec)
    model = type("M", (), {"update": 100})()
    trainer.window_peak_memory("cuda")
    image.image.get_current_log(model)
    image.image.get_current_log(model)  # no new peak read: no mem_gib
    rows = [json.loads(x) for x in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert rows[0]["mem_gib"] == pytest.approx(4.7)
    assert rows[0]["l_g_pix"] == pytest.approx(1.0)
    assert "mem_gib" not in rows[1]
