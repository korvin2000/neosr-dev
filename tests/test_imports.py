"""Importing a neosr module has no side effects: no argv parsing, CUDA init, default device or files."""

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
    "neosr.config",
    "neosr.cli",
    "neosr.runtime",
    "neosr.utils",
    "neosr.utils.factory",
    "neosr.utils.dist_util",
    "neosr.losses",
    "neosr.metrics",
    "neosr.optimizers",
    *sorted(XFAIL),
]

# Parent packages that are still in XFAIL are replaced by empty namespace stubs, so that only the module's
# own import is measured. A package also imports all its submodules (a package may swallow their errors).
# Arguments travel by environment, and any argparse read of sys.argv is recorded.
PROBE = """
import argparse, importlib, json, os, pkgutil, sys, types
repo, module = os.environ["PROBE_REPO"], os.environ["PROBE_MODULE"]
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
for sub in pkgutil.walk_packages(getattr(mod, "__path__", []), module + "."):
    importlib.import_module(sub.name)
import torch
print(json.dumps({"file": mod.__file__, "cuda": torch.cuda.is_initialized(), "argv": argv_reads,
                  "device": str(torch.get_default_device())}))
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
    assert not any(tmp_path.rglob("*")), "import wrote files into the working directory"
