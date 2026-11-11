"""scripts/ab/analyze.py on synthetic runs: the cluster bootstrap, the gate and the data guards. CPU."""

import json

import numpy as np
import pytest

from scripts.ab.analyze import Data, bootstrap, main, paired, status, verdict

IT, N_GROUPS, PER_GROUP = 8000, 20, 5
BASE = {"psnr": 30.0, "topiq": 0.65, "lpips": 0.17, "dists": 0.15}


def _ids() -> list[str]:
    return [f"g{g:02d}i{i}" for g in range(N_GROUPS) for i in range(PER_GROUP)]


def _data(tmp, exclude=()) -> None:
    (tmp / "dev_gt").mkdir()
    for i in _ids():
        (tmp / "dev_gt" / f"{i}.png").touch()
    groups = [
        [f"{i}.png" for i in _ids() if i.startswith(f"g{g:02d}")]
        for g in range(N_GROUPS)
    ]
    (tmp / "sources.json").write_text(json.dumps(groups))
    (tmp / "exclude.txt").write_text("\n".join(exclude))


def _arm(runs, name, shift=None, seed=0, drop=0) -> None:
    """Per-image rows: a fixed per-image base (shared by every arm), small noise, plus `shift`."""
    base = np.random.default_rng(99)
    noise = np.random.default_rng(seed)
    (runs / name).mkdir(parents=True)
    rows = []
    for i in _ids()[drop:]:
        row = {"iter": IT, "dataset": "dev", "id": i}
        for m, v in BASE.items():
            scale = 0.5 if m == "psnr" else 0.01
            row[m] = (
                v
                + scale * base.normal()
                + 1e-4 * noise.normal()
                + (shift or {}).get(m, 0.0)
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
