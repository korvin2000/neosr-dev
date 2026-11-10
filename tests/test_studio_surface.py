"""The Studio contract: studio/shared/catalog/neosr-surface.json mirrors the trainer's tables.

A failure means a table changed (ARCHS, LOSSES, METRICS, OPTIMIZERS, SCHEDULERS, config.SCHEMA, the
gan types or convert.py's export tables): run `python scripts/studio_surface.py`, commit the JSON and
update the Studio catalog its own contract test (`studio/src/app/core/catalog-contract.spec.ts`) names.
"""

import importlib.util
import json
from pathlib import Path

from neosr.archs import ARCHS
from neosr.losses import LOSSES
from neosr.metrics import METRICS
from neosr.models.base import OPTIMIZERS, SCHEDULERS

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "neosr_studio_surface", ROOT / "scripts" / "studio_surface.py"
)
surface = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
spec.loader.exec_module(surface)  # type: ignore[union-attr]


def test_surface_file_is_current():
    assert json.loads(surface.SURFACE.read_text("utf-8")) == surface.build()


def test_static_read_matches_the_imported_tables():
    built = surface.build()
    archs = {*built["generators"], *built["discriminators"], *surface.INTERNAL_ARCHS}
    assert archs == set(ARCHS)
    assert set(built["losses"]) == set(LOSSES)
    assert set(built["metrics"]) == set(METRICS)
    assert set(built["optimizers"]) == set(OPTIMIZERS)
    assert set(built["schedulers"]) == set(SCHEDULERS)


def test_shipped_templates_pass_the_validator(capsys):
    templates = sorted(str(p) for p in (ROOT / "options").glob("train_*.toml"))
    assert templates
    assert surface.validate(templates) == 0, capsys.readouterr().out
