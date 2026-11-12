"""The trainer's configuration surface as JSON: the contract NeoSR Studio's catalog is tested against.

Reads the registries through their own source scan and the remaining tables statically (ast),
the constructor signatures of archs, losses, metrics, optimizers and schedulers included (`*_params`);
`neosr/config.py` and `neosr/utils/registry.py` are loaded by path (both are torch-free), so it runs
on any Python >= 3.12 without torch, CUDA or the project environment.

    python scripts/studio_surface.py                  # rewrite studio/shared/catalog/neosr-surface.json
    python scripts/studio_surface.py --check          # exit 1 when that file is stale
    python scripts/studio_surface.py --validate a.toml b.toml
        # load_config + validate_train on training configs (e.g. Studio output); an unknown key fails

`--validate` also checks the opaque sections (network_*, train.*_opt, optimizers, scheduler,
metrics) against the constructor signatures: an unknown type, a keyword the constructor does not
take (unless it has **kwargs) or one a preset binds fails. Constructors outside the repository
(torch.optim, torch.optim.lr_scheduler) are recorded as `external` and not checked.
"""

import argparse
import ast
import functools
import importlib.util
import json
import sys
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SURFACE = ROOT / "studio" / "shared" / "catalog" / "neosr-surface.json"
INTERNAL_ARCHS = {"VGGFeatureExtractor"}  # scale-free but not a discriminator
# a class without an __init__ of its own on these bases takes no keywords
PARAMLESS_BASES = {"Loss", "nn.Module"}
# keys a builder reads itself or passes on its own, besides `type` (neosr/losses/__init__.py
# build_losses, neosr/models/base.py build_schedulers, neosr/metrics/__init__.py build_metric)
BUILDER_KEYS = {
    "loss_params": ("loss_weight", "perceptual_weight", "target"),
    "scheduler_params": ("nets",),
    "metric_params": ("better",),
}
TYPE_NAMES = {dict: "table", list: "array"}


def _load(rel: str) -> Any:
    """A torch-free neosr module by path: importing its package would import torch."""
    spec = importlib.util.spec_from_file_location(Path(rel).stem, ROOT / rel)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@functools.cache
def _tree(rel: str) -> ast.Module:
    return ast.parse((ROOT / rel).read_text("utf-8"), filename=rel)


@functools.cache
def _defs(rel: str) -> dict[str, ast.stmt]:
    return {
        n.name: n
        for n in _tree(rel).body
        if isinstance(n, ast.ClassDef | ast.FunctionDef)
    }


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
            if node.bases and all(
                ast.unparse(b) in PARAMLESS_BASES for b in node.bases
            ):
                return Signature(ast.parse("def f(): pass").body[0].args)  # type: ignore[attr-defined]
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


def _constructor(rel: str, name: str, drop: tuple[str, ...] = ()) -> dict[str, Any]:
    """`name`'s signature without the arguments its builder passes itself (`drop`)."""
    sig = _signature(_defs(rel), name, rel)
    for key in drop:
        sig.params.pop(key, None)
    return sig.as_json()


def _registry_params(
    registry: Any, package: str, suffix: str = "", drop: tuple[str, ...] = ()
) -> dict[str, Any]:
    return {
        name: _constructor(f"neosr/{package}/{row.module}.py", name + suffix, drop)
        for name, row in sorted(registry.items())
    }


def _origin(rel: str, name: str) -> tuple[str, str]:
    """(file, name) of the module-level class or def that `name` is in `rel`, through
    `from neosr... import` re-exports; ("", dotted path) for one from outside the repository."""
    head, _, rest = name.partition(".")
    if not rest and head in _defs(rel):
        return rel, head
    for node in _tree(rel).body:
        if not isinstance(node, ast.Import | ast.ImportFrom):
            continue
        for alias in node.names:
            if (alias.asname or alias.name.partition(".")[0]) != head:
                continue
            if isinstance(node, ast.Import):  # `import a.b` binds `a`
                full, module = alias.name if alias.asname else head, None
            else:
                full, module = f"{node.module}.{alias.name}", node.module
            if not full.startswith("neosr."):
                return "", ".".join(filter(None, (full, rest)))
            if module is None or rest:
                break  # an attribute of a neosr module: not a form the tables use
            path = module.replace(".", "/")
            sub = (
                f"{path}.py"
                if (ROOT / f"{path}.py").exists()
                else f"{path}/__init__.py"
            )
            return _origin(sub, alias.name)
    msg = f"{rel}: cannot resolve {name}"
    raise LookupError(msg)


def _table_params(rel: str, table: str, drop: tuple[str, ...]) -> dict[str, Any]:
    """The constructors of a `{type: class}` table; one outside the repository is `external`."""
    node = _assigned(rel, table)
    if not isinstance(node, ast.Dict):
        msg = f"{rel}: {table} is not a dict literal"
        raise TypeError(msg)
    out = {}
    for key, value in zip(node.keys, node.values, strict=True):
        where, name = _origin(rel, ast.unparse(value))
        out[ast.literal_eval(key)] = (  # type: ignore[arg-type]
            _constructor(where, name, drop) if where else {"external": name}
        )
    return dict(sorted(out.items()))


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
        "arch_params": _registry_params(archs, "archs"),
        "losses": sorted(reg.LOSS_REGISTRY),
        "loss_params": _registry_params(reg.LOSS_REGISTRY, "losses"),
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
        "metric_params": _registry_params(  # the bound twin `<type>_pt(img, img2, ...)`
            reg.METRIC_REGISTRY, "metrics", "_pt", ("img", "img2", "device")
        ),
        "metric_lower": sorted(cfg.METRIC_LOWER),
        "optimizers": sorted(_keys("neosr/models/base.py", "OPTIMIZERS")),
        "optimizer_params": _table_params(
            "neosr/models/base.py", "OPTIMIZERS", ("params",)
        ),
        "schedulers": sorted(_keys("neosr/models/base.py", "SCHEDULERS")),
        "scheduler_params": _table_params(
            "neosr/models/base.py", "SCHEDULERS", ("optimizer",)
        ),
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


def _opaque_tables(opt: dict[str, Any]) -> list[tuple[str, Any, str]]:
    """(section, table, surface key) of every constructor-checked table in a loaded config."""
    train, val = opt.get("train") or {}, opt.get("val") or {}
    return [
        *((key, opt.get(key), "arch_params") for key in ("network_g", "network_d")),
        *(
            (f"train.{k}", v, "loss_params")
            for k, v in train.items()
            if k.endswith("_opt")
        ),
        *(
            (f"train.{k}", train.get(k), "optimizer_params")
            for k in ("optim_g", "optim_d")
        ),
        ("train.scheduler", train.get("scheduler"), "scheduler_params"),
        *(
            (f"val.metrics.{k}", v, "metric_params")
            for k, v in (val.get("metrics") or {}).items()
        ),
    ]


def kwarg_problems(opt: dict[str, Any], surface: dict[str, Any]) -> list[str]:
    """Unknown types, and keywords a constructor rejects (unknown without **kwargs, or preset-bound)."""
    own = {
        **BUILDER_KEYS,
        "loss_params": BUILDER_KEYS["loss_params"]
        + tuple(surface["loss_schedule_keys"]),
    }
    bad = []
    for section, table, key in _opaque_tables(opt):
        if not isinstance(table, dict) or not table:
            continue  # absent, or an empty loss table the builder skips
        kind = table.get("type")
        if key in {"optimizer_params", "scheduler_params"}:
            kind = str(kind).lower()  # looked up lower-cased
        sig = surface[key].get(kind)
        if sig is None:
            bad.append(f"[{section}] unknown type {kind!r}")
            continue
        if "external" in sig or sig.get("var_kwargs"):
            continue
        fixed = set(sig.get("fixed", ()))
        for k in table.keys() - {"type", *own.get(key, ())}:
            if k in fixed or k not in sig["params"]:
                why = (
                    "set by the preset" if k in fixed else "not a constructor parameter"
                )
                bad.append(f"[{section}] {kind}: {k!r} is {why}")
    return bad


def _validate_one(cfg: Any, surface: dict[str, Any], path: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", cfg.ConfigWarning)
        opt = cfg.load_config(path, is_train=True, root=ROOT)
        cfg.validate_train(opt)
    if bad := kwarg_problems(opt, surface):
        raise cfg.ConfigError("; ".join(bad))


def validate(paths: list[str]) -> int:
    """load_config + validate_train + the constructor keywords per file; a ConfigWarning (unknown
    key) counts as a failure."""
    cfg, surface, failed = _load("neosr/config.py"), build(), 0
    for path in paths:
        try:
            _validate_one(cfg, surface, path)
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
