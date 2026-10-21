"""Importing a neosr module has no side effects: no argv parsing, CUDA init, default device, RNG or files."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Not yet migrated: still parse argv, scan folders or star-import at import time. Shrinks as lanes land;
# strict xfail makes a migrated module fail here until it is removed from the set.
XFAIL: set[str] = set()
MODULES = [
    "neosr",
    "neosr.archs",
    "neosr.data",
    "neosr.config",
    "neosr.cli",
    "neosr.runtime",
    "neosr.utils",
    "neosr.utils.factory",
    "neosr.utils.dist_util",
    "neosr.losses",
    "neosr.metrics",
    "neosr.models",
    "neosr.optimizers",
    *sorted(XFAIL),
]

# Lazy builder tables (D3): importing the package must not import the modules its rows name.
TABLES = {
    "neosr.archs": "ARCHS",
    "neosr.data": "DATASETS",
    "neosr.losses": "LOSSES",
    "neosr.metrics": "METRICS",
    "neosr.models": "MODELS",
}

# Parent packages that are still in XFAIL are replaced by empty namespace stubs, so that only the module's
# own import is measured. A package also imports all its submodules (a package may swallow their errors).
# Arguments travel by environment, and any argparse read of sys.argv is recorded.
PROBE = """
import argparse, importlib, json, os, pkgutil, random, sys, types
import numpy as np, torch
repo, module, table = (os.environ[k] for k in ("PROBE_REPO", "PROBE_MODULE", "PROBE_TABLE"))
def rng_state():
    return [repr(random.getstate()), np.random.get_state()[1].tolist(), torch.get_rng_state().tolist()]
rng_before = rng_state()
for name in filter(None, os.environ["PROBE_STUBS"].split(",")):
    stub = types.ModuleType(name)
    stub.__path__ = [os.path.join(repo, *name.split("."))]
    sys.modules[name] = stub
argv_reads = []
parse_known_args = argparse.ArgumentParser.parse_known_args
def spy(self, args=None, namespace=None):
    if args is None:
        argv_reads.append(self.prog)
    return parse_known_args(self, args, namespace)
argparse.ArgumentParser.parse_known_args = spy
mod = importlib.import_module(module)
rows = {f"{module}.{row[0]}" for row in getattr(mod, table).values()} if table else set()
eager = sorted(rows & sys.modules.keys())
for sub in pkgutil.walk_packages(getattr(mod, "__path__", []), module + "."):
    importlib.import_module(sub.name)
print(json.dumps({"file": mod.__file__, "cuda": torch.cuda.is_initialized(), "argv": argv_reads,
                  "device": str(torch.get_default_device()), "eager": eager,
                  "rng": rng_state() == rng_before}))
"""


def probe(module: str, cwd: Path) -> subprocess.CompletedProcess:
    parts = module.split(".")
    stubs = [
        p for p in (".".join(parts[:i]) for i in range(1, len(parts))) if p in XFAIL
    ]
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PROBE_REPO": str(ROOT),
        "PROBE_MODULE": module,
        "PROBE_STUBS": ",".join(stubs),
        "PROBE_TABLE": TABLES.get(module, ""),
    }
    return subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


@pytest.mark.parametrize(
    "module",
    [
        pytest.param(m, marks=pytest.mark.xfail(strict=True, reason="not migrated"))
        if m in XFAIL
        else m
        for m in MODULES
    ],
)
def test_import_has_no_side_effects(module: str, tmp_path: Path) -> None:
    result = probe(module, tmp_path)
    assert result.returncode == 0, result.stderr[-2000:]
    info = json.loads(result.stdout.strip().splitlines()[-1])
    assert Path(info["file"]).resolve().is_relative_to(ROOT / "neosr")
    assert not info["argv"], "import read sys.argv"
    assert info["cuda"] is False
    assert info["device"] == "cpu"
    assert info["eager"] == [], "package imported its table modules eagerly"
    assert info["rng"], "import changed the random, numpy or torch RNG state"
    assert not any(tmp_path.rglob("*")), "import wrote files into the working directory"
