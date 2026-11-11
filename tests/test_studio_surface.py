"""The Studio contract: studio/shared/catalog/neosr-surface.json mirrors the trainer's tables.

A failure means a registry or table changed (the five registries, an arch constructor, OPTIMIZERS, SCHEDULERS,
config.SCHEMA, the gan types or convert.py's export tables): run `python scripts/studio_surface.py`, commit the JSON and
update the Studio catalog its own contract test (`studio/src/app/core/catalog-contract.spec.ts`) names.
"""

import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from neosr.models.base import OPTIMIZERS, SCHEDULERS
from neosr.utils.registry import ARCH_REGISTRY, LOSS_REGISTRY, METRIC_REGISTRY

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "neosr_studio_surface", ROOT / "scripts" / "studio_surface.py"
)
surface = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
spec.loader.exec_module(surface)  # type: ignore[union-attr]


def test_surface_file_is_current():
    assert json.loads(surface.SURFACE.read_text("utf-8")) == surface.build()


def test_surface_script_needs_no_torch():
    """The script loads registry.py and config.py by path; importing neosr.utils would load torch."""
    code = (
        "import runpy, sys\n"
        "sys.modules.update(dict.fromkeys(('torch', 'numpy', 'cv2')))\n"
        "sys.argv = ['studio_surface.py', '--check']\n"
        f"runpy.run_path({str(ROOT / 'scripts' / 'studio_surface.py')!r}, run_name='__main__')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_static_read_matches_the_imported_tables():
    built = surface.build()
    archs = {*built["generators"], *built["discriminators"], *surface.INTERNAL_ARCHS}
    assert archs == set(ARCH_REGISTRY)
    assert set(built["losses"]) == set(LOSS_REGISTRY)
    assert set(built["metrics"]) == set(METRIC_REGISTRY)
    assert set(built["optimizers"]) == set(OPTIMIZERS)
    assert set(built["schedulers"]) == set(SCHEDULERS)


def test_shipped_templates_pass_the_validator(capsys):
    templates = sorted(str(p) for p in (ROOT / "options").glob("train_*.toml"))
    assert templates
    assert surface.validate(templates) == 0, capsys.readouterr().out


def _jsonable(value):
    return [_jsonable(v) for v in value] if isinstance(value, tuple | list) else value


def test_arch_signatures_match_the_imported_constructors():
    """The ast read of each arch constructor (`arch_params`) against inspect on the real object;
    a preset's fixed keys raise TypeError (bound twice) before anything is built."""
    built = surface.build()["arch_params"]
    assert set(built) == set(ARCH_REGISTRY)
    for name in ARCH_REGISTRY:
        sig, obj = built[name], ARCH_REGISTRY.load(name)
        params = inspect.signature(obj).parameters
        if inspect.isclass(obj):
            named = {k: p for k, p in params.items() if p.kind is not p.VAR_KEYWORD}
            assert list(sig["params"]) == list(named), name
            assert sig.get("var_kwargs", False) == any(
                p.kind is p.VAR_KEYWORD for p in params.values()
            ), name
            for key, p in named.items():
                if key in sig.get("required", ()):
                    assert p.default is p.empty, (name, key)
                elif key not in sig.get("nonliteral", ()):
                    assert sig["params"][key] == _jsonable(p.default), (name, key)
        for key in sig.get("fixed", ()):
            with pytest.raises(TypeError, match="multiple values"):
                obj(**{key: None})
