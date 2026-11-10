"""The Studio contract: studio/shared/catalog/neosr-surface.json mirrors the trainer's tables.

A failure means a registry or table changed (the five registries, OPTIMIZERS, SCHEDULERS, config.SCHEMA, the
gan types or convert.py's export tables): run `python scripts/studio_surface.py`, commit the JSON and
update the Studio catalog its own contract test (`studio/src/app/core/catalog-contract.spec.ts`) names.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

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
