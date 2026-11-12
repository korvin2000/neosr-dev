"""scripts/ab/queue.sh with a stub trainer: a line it cannot run stops the queue and stays in the
file; the flock on the queue file keeps a second queue out. CPU, bash."""

import fcntl
import os
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("flock") is None, reason="bash + flock"
)

QUEUE = Path(__file__).resolve().parents[1] / "scripts" / "ab" / "queue.sh"
# run_ab.py stand-in: $2 code, $3 toml, $4 out. Arm a starts a second queue on the same file.
STUB = """#!/bin/bash
out=$4
[ "$(basename "$out")" = a ] && { bash "$QUEUE_SH" > "$out/second.log" 2>&1; echo $? > "$out/second_rc"; }
echo '{}' > "$out/done.json"
"""


def _run(tmp: Path, lines: str) -> subprocess.CompletedProcess:
    (tmp / "cfg").mkdir(exist_ok=True)
    for arm in ("a", "b", "c"):
        (tmp / "cfg" / f"{arm}.toml").touch()
    stub = tmp / "stub.sh"
    stub.write_text(STUB)
    stub.chmod(0o755)
    (tmp / "q.txt").write_text(lines)
    env = os.environ | {
        "CFG": str(tmp / "cfg"),
        "RUNS": str(tmp / "runs"),
        "QUEUE": str(tmp / "q.txt"),
        "PY": str(stub),
        "QUEUE_SH": str(QUEUE),
        "DEADLINE": "23:59",
        "SINGLE_MIN": "0",
        "PAIR_MIN": "0",
        "MARGIN_MIN": "0",
    }
    env.pop("LOCK", None)
    env.pop("AB_LOCKED", None)
    return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [shutil.which("bash"), str(QUEUE)],
        cwd=tmp,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


@pytest.mark.parametrize(
    "bad", ["B0 B0 resume", "a b", "x", "a,x", "a,b,c", "a,a", ",b", "../a"]
)
def test_invalid_line_stops_and_stays(tmp_path, bad):
    r = _run(tmp_path, f"# note\nb\n{bad}\nc\n")
    assert r.returncode == 1
    assert f"STOP at '{bad}'" in r.stderr
    assert (tmp_path / "q.txt").read_text() == f"{bad}\nc\n"  # nothing past it consumed
    assert (tmp_path / "runs" / "b" / "done.json").exists()
    assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == ["b"]


def test_pair_and_blanks_run_and_a_second_queue_refuses(tmp_path):
    """The lock survives the in-place line removal: arm a, started after its line left the file,
    tries a second queue on the same file, which refuses and runs nothing."""
    r = _run(tmp_path, " a , b \r\n\nc\n")
    assert r.returncode == 0, r.stderr
    assert "start a b" in r.stdout
    assert "end c rc=0 done=yes" in r.stdout
    assert (tmp_path / "runs" / "a" / "second_rc").read_text().strip() == "2"
    assert (
        "run by another queue" in (tmp_path / "runs" / "a" / "second.log").read_text()
    )
    assert not (tmp_path / "q.txt").read_text()


def test_held_queue_file_refuses_to_start(tmp_path):
    (tmp_path / "q.txt").write_text("")
    with (tmp_path / "q.txt").open() as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        r = _run(tmp_path, "a\n")
    assert r.returncode == 2
    assert "run by another queue" in r.stderr
    assert (tmp_path / "q.txt").read_text() == "a\n"
    assert not (tmp_path / "runs").exists()
