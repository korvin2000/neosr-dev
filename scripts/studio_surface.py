"""The trainer's configuration surface as JSON: the contract NeoSR Studio's catalog is tested against.

Reads the tables statically (ast; `neosr/config.py` is imported, it is torch-free), so it runs on any
Python >= 3.11 without torch, CUDA or the project environment.

    python scripts/studio_surface.py                  # rewrite studio/shared/catalog/neosr-surface.json
    python scripts/studio_surface.py --check          # exit 1 when that file is stale
    python scripts/studio_surface.py --validate a.toml b.toml
        # load_config + validate_train on training configs (e.g. Studio output); an unknown key fails

Opaque sections (network_*, train.*_opt, optimizers, scheduler, metrics) are checked by their
constructors, which need torch: `--validate` covers the schema and the cross-key rules only.
"""

import argparse
import ast
import importlib.util
import json
import sys
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SURFACE = ROOT / "studio" / "shared" / "catalog" / "neosr-surface.json"
INTERNAL_ARCHS = {"VGGFeatureExtractor"}  # scale-free but not a discriminator
TYPE_NAMES = {dict: "table", list: "array"}


def _config() -> Any:
    spec = importlib.util.spec_from_file_location(
        "neosr_config", ROOT / "neosr" / "config.py"
    )
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _tree(rel: str) -> ast.Module:
    return ast.parse((ROOT / rel).read_text("utf-8"), filename=rel)


def _assigned(rel: str, name: str, *, nested: bool = False) -> ast.expr:
    """The value assigned to `name` in `rel` (module level, or anywhere with `nested`)."""
    tree = _tree(rel)
    for node in ast.walk(tree) if nested else tree.body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return node.value  # type: ignore[union-attr,return-value]
    msg = f"{rel}: no assignment to {name}"
    raise LookupError(msg)


def _table(rel: str, name: str) -> dict[str, Any]:
    return ast.literal_eval(_assigned(rel, name))


def _keys(rel: str, name: str, *, nested: bool = False) -> list[str]:
    node = _assigned(rel, name, nested=nested)
    if not isinstance(node, ast.Dict):
        msg = f"{rel}: {name} is not a dict literal"
        raise TypeError(msg)
    return [ast.literal_eval(k) for k in node.keys if k is not None]


def _gan_types() -> list[str]:
    """gan_loss: the criterion table plus the label-free types of the `{*crits, ...}` check."""
    rel = "neosr/losses/gan_loss.py"
    kinds = set(_keys(rel, "crits", nested=True))
    for node in ast.walk(_tree(rel)):
        if isinstance(node, ast.Set) and any(
            isinstance(e, ast.Starred) and getattr(e.value, "id", None) == "crits"
            for e in node.elts
        ):
            kinds |= {e.value for e in node.elts if isinstance(e, ast.Constant)}
            return sorted(kinds)
    msg = f"{rel}: the gan_type check was not found"
    raise LookupError(msg)


def _types(expected: Any) -> str | list[str]:
    if isinstance(expected, tuple):
        return [TYPE_NAMES.get(t, t.__name__) for t in expected]
    return TYPE_NAMES.get(expected, expected.__name__)


def build() -> dict[str, Any]:
    cfg = _config()
    archs = _table("neosr/archs/__init__.py", "ARCHS")
    scale_free = {k for k, (_, scale) in archs.items() if scale is None}
    return {
        "generators": sorted(archs.keys() - scale_free),
        "discriminators": sorted(scale_free - INTERNAL_ARCHS),
        "scale_kwarg": {k: v[1] for k, v in sorted(archs.items()) if v[1]},
        "losses": sorted(_table("neosr/losses/__init__.py", "LOSSES")),
        "loss_schedule_keys": list(
            ast.literal_eval(_assigned("neosr/losses/__init__.py", "SCHEDULE_KEYS"))
        ),
        "gan_types": _gan_types(),
        "metrics": sorted(_table("neosr/metrics/__init__.py", "METRICS")),
        "metric_choices": {
            metric: {key: list(values) for key, values in choices.items()}
            for metric, choices in _table(
                "neosr/metrics/__init__.py", "CHOICES"
            ).items()
        },
        "metric_lower": sorted(cfg.METRIC_LOWER),
        "optimizers": sorted(_keys("neosr/models/base.py", "OPTIMIZERS")),
        "schedulers": sorted(_keys("neosr/models/base.py", "SCHEDULERS")),
        "models": sorted(_table("neosr/models/__init__.py", "MODELS")),
        "datasets": sorted(_table("neosr/data/__init__.py", "DATASETS")),
        "scales": sorted(cfg.SCALES),
        "required": list(cfg.REQUIRED),
        "schema": {
            section: {key: _types(t) for key, t in sorted(keys.items())}
            for section, keys in cfg.SCHEMA.items()
        },
        "opaque": list(cfg.OPAQUE),
        "removed": dict(sorted(cfg.REMOVED.items())),
        "no_onnx": sorted(_keys("convert.py", "NO_ONNX")),
        "onnx_static_only": sorted(_keys("convert.py", "STATIC_ONLY")),
    }


def validate(paths: list[str]) -> int:
    """load_config + validate_train per file; a ConfigWarning (unknown key) counts as a failure."""
    cfg, failed = _config(), 0
    for path in paths:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", cfg.ConfigWarning)
                cfg.validate_train(cfg.load_config(path, is_train=True, root=ROOT))
        except (cfg.ConfigError, cfg.ConfigWarning) as err:
            failed += 1
            print(f"FAIL {path}: {err}")
        else:
            print(f"ok   {path}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0]  # type: ignore[union-attr]
    )
    parser.add_argument(
        "--check", action="store_true", help="fail when the JSON is stale"
    )
    parser.add_argument(
        "--validate", nargs="+", metavar="TOML", help="validate training configs"
    )
    args = parser.parse_args()
    if args.validate:
        return validate(args.validate)
    surface = build()
    if args.check:
        if not SURFACE.exists() or json.loads(SURFACE.read_text("utf-8")) != surface:
            print(
                f"{SURFACE.relative_to(ROOT)} is stale: run scripts/studio_surface.py"
            )
            return 1
        print(f"{SURFACE.relative_to(ROOT)} is current")
        return 0
    SURFACE.write_text(
        json.dumps(surface, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {SURFACE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
