"""The trainer's configuration surface as JSON: the contract NeoSR Studio's catalog is tested against.

Reads the registries through their own source scan and the remaining tables statically (ast),
the arch constructor signatures included (`arch_params`);
`neosr/config.py` and `neosr/utils/registry.py` are loaded by path (both are torch-free), so it runs
on any Python >= 3.12 without torch, CUDA or the project environment.

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


def _load(rel: str) -> Any:
    """A torch-free neosr module by path: importing its package would import torch."""
    spec = importlib.util.spec_from_file_location(Path(rel).stem, ROOT / rel)
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


NONLITERAL = object()


def _default(node: ast.expr) -> Any:
    try:
        return ast.literal_eval(node)
    except ValueError:
        return NONLITERAL


def _jsonable(value: Any) -> Any:
    return [_jsonable(v) for v in value] if isinstance(value, tuple | list) else value


class Signature:
    """Constructor parameters: name -> literal default (`NONLITERAL`, or `REQUIRED` without one)."""

    REQUIRED = object()

    def __init__(self, args: ast.arguments, *, skip_self: bool = False) -> None:
        if args.vararg is not None or args.posonlyargs:
            msg = "positional-only or *args parameters cannot be configured from TOML"
            raise TypeError(msg)
        positional = args.args[1:] if skip_self else args.args
        defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
        pairs = [
            *zip(positional, defaults, strict=True),
            *zip(args.kwonlyargs, args.kw_defaults, strict=True),
        ]
        self.params: dict[str, Any] = {
            a.arg: self.REQUIRED if d is None else _default(d) for a, d in pairs
        }
        self.fixed: set[str] = set()  # bound by a preset: passing it raises TypeError
        self.var_kwargs = args.kwarg is not None

    def as_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "params": {
                k: None if v is NONLITERAL or v is self.REQUIRED else _jsonable(v)
                for k, v in self.params.items()
            }
        }
        lists = {
            "required": [k for k, v in self.params.items() if v is self.REQUIRED],
            "nonliteral": [k for k, v in self.params.items() if v is NONLITERAL],
            "fixed": sorted(self.fixed),
        }
        out |= {key: names for key, names in lists.items() if names}
        if self.var_kwargs:
            out["var_kwargs"] = True
        return out


def _signature(defs: dict[str, ast.stmt], name: str, rel: str) -> Signature:
    """A class's `__init__`; a preset `def p(**kwargs): return base(k=v, ..., **kwargs)` resolves
    to the base's parameters with the keywords it passes as fixed, plus its own parameters, which
    stay open (a renamed or derived one: the base key is fixed, the preset's name is open)."""
    node = defs.get(name)
    if isinstance(node, ast.ClassDef):
        init = next(
            (
                b
                for b in node.body
                if isinstance(b, ast.FunctionDef) and b.name == "__init__"
            ),
            None,
        )
        if init is None:
            msg = f"{rel}: {name} has no __init__ of its own"
            raise LookupError(msg)
        return Signature(init.args, skip_self=True)
    if not isinstance(node, ast.FunctionDef):
        msg = f"{rel}: {name} is not a module-level class or def"
        raise TypeError(msg)
    own = Signature(node.args)
    body = [
        b
        for b in node.body
        if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant))
    ]
    call = body[0].value if len(body) == 1 and isinstance(body[0], ast.Return) else None
    if not (
        own.var_kwargs
        and isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and not call.args
        and any(k.arg is None for k in call.keywords)
    ):
        return own  # not a preset: its own parameters are the surface
    sig = _signature(defs, call.func.id, rel)
    for kw in call.keywords:
        if kw.arg is None:
            continue
        # every keyword the preset passes is bound (a TOML value for it raises TypeError), unless
        # it forwards its own parameter of the same name, which the loop below opens again
        name_ = kw.value.id if isinstance(kw.value, ast.Name) else None
        sig.params[kw.arg] = (
            own.params[name_] if name_ in own.params else _default(kw.value)
        )
        sig.fixed.add(kw.arg)
    for (
        key,
        default,
    ) in own.params.items():  # the preset's own parameters are what TOML sets
        sig.params[key] = default
        sig.fixed.discard(key)
    return sig


def _arch_params(archs: Any) -> dict[str, Any]:
    trees: dict[str, dict[str, ast.stmt]] = {}
    out = {}
    for name, row in sorted(archs.items()):
        rel = f"neosr/archs/{row.module}.py"
        if rel not in trees:
            trees[rel] = {
                n.name: n
                for n in _tree(rel).body
                if isinstance(n, ast.ClassDef | ast.FunctionDef)
            }
        out[name] = _signature(trees[rel], name, rel).as_json()
    return out


def _types(expected: Any) -> str | list[str]:
    if isinstance(expected, tuple):
        return [TYPE_NAMES.get(t, t.__name__) for t in expected]
    return TYPE_NAMES.get(expected, expected.__name__)


def build() -> dict[str, Any]:
    cfg, reg = _load("neosr/config.py"), _load("neosr/utils/registry.py")
    archs = reg.ARCH_REGISTRY
    scale_free = {k for k, row in archs.items() if row.scale is None}
    return {
        "generators": sorted(archs.keys() - scale_free),
        "discriminators": sorted(scale_free - INTERNAL_ARCHS),
        "scale_kwarg": {k: row.scale for k, row in sorted(archs.items()) if row.scale},
        "arch_params": _arch_params(archs),
        "losses": sorted(reg.LOSS_REGISTRY),
        "loss_schedule_keys": list(
            ast.literal_eval(_assigned("neosr/losses/__init__.py", "SCHEDULE_KEYS"))
        ),
        "gan_types": _gan_types(),
        "metrics": sorted(reg.METRIC_REGISTRY),
        "metric_choices": {
            metric: row.meta["choices"]
            for metric, row in reg.METRIC_REGISTRY.items()
            if "choices" in row.meta
        },
        "metric_lower": sorted(cfg.METRIC_LOWER),
        "optimizers": sorted(_keys("neosr/models/base.py", "OPTIMIZERS")),
        "schedulers": sorted(_keys("neosr/models/base.py", "SCHEDULERS")),
        "models": sorted(reg.MODEL_REGISTRY),
        "datasets": sorted(reg.DATASET_REGISTRY),
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
    cfg, failed = _load("neosr/config.py"), 0
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
