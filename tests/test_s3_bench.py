"""Stage 3 A5: scripts/bench.py smoke tests (CPU, eager, a tiny net; the GPU runs are the owner's)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from neosr.models.image import image
from tests.test_entry import _restore_globals as _restore_globals  # autouse fixture
from tests.test_entry import cpu, data, toml  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "neosr_bench", ROOT / "scripts" / "bench.py"
)
bench = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
spec.loader.exec_module(bench)  # type: ignore[union-attr]


def _run(capsys, cfg, *extra):
    argv = [
        "-opt",
        str(cfg),
        "--trials",
        "2",
        "--updates",
        "3",
        "--warmup",
        "2",
        "--infer",
        "16",
    ]
    assert bench.main([*argv, *extra]) == 0
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


@pytest.mark.usefixtures("cpu")
def test_bench_reports_steady_state_and_writes_nothing(tmp_path, data, capsys):  # noqa: F811
    out = _run(
        capsys, toml(tmp_path, data, total=1000), "--eager", "--precision", "fp32"
    )
    assert out["device"] == "cpu"
    assert (
        out["compile"],
        out["precision"],
        out["trials"],
        out["updates_per_trial"],
    ) == (False, "fp32", 2, 3)
    assert out["ms_per_update"] > 0
    lo, hi = out["ms_per_update_min_max"]
    assert lo <= out["ms_per_update"] <= hi
    assert out["first_update_s"] > 0
    assert out["warmup_updates"] == 2
    assert out["peak_vram_train_mib"] is None  # CPU
    if sys.platform != "win32":
        assert out["peak_host_rss_mib"] > 0
    assert (out["infer_size"], out["infer_ms"] > 0) == (16, True)
    assert not (tmp_path / "exp").exists()  # no run folder, state, export or log


@pytest.mark.usefixtures("cpu")
def test_bench_drives_the_loop_across_epochs_and_logs_by_update(
    tmp_path, data, capsys, monkeypatch
):  # noqa: F811
    """6 images, batch 2, accumulate 2: 1.5 updates per epoch; print_freq 3 reads at updates 3 and 6."""
    cfg = toml(tmp_path, data, total=1000, accumulate=2)
    cfg.write_text(cfg.read_text().replace("print_freq = 1", "print_freq = 3"))
    reads, real = [], image.get_current_log
    monkeypatch.setattr(
        image, "get_current_log", lambda self: reads.append(self.update) or real(self)
    )
    _run(capsys, cfg)
    assert reads == [3, 6]  # 2 warm-up + 2 x 3 timed updates


def test_bench_flags():
    args = bench.parse([
        "-opt",
        "x.toml",
        "--compile",
        "--compile_mode",
        "max-autotune",
    ])
    assert (args.compile, args.compile_mode, args.precision, args.warmup) == (
        True,
        "max-autotune",
        None,
        None,
    )
    assert bench.parse(["-opt", "x.toml"]).compile is None  # the config decides
    assert bench._precision({"use_amp": True, "bfloat16": True}) == "bf16"
    assert bench._precision({}) == "fp32"
    for bad in (["--updates", "0"], ["--warmup", "0"], ["--trials", "0"]):
        with pytest.raises(SystemExit):
            bench.parse(["-opt", "x.toml", *bad])
