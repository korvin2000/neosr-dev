"""The five registries. `@X_REGISTRY.register()` above a class or function declares a type.

A registry never imports its package to learn what it holds: on first use it reads the decorators
from the package's source text and is from then on a read-only mapping `type -> Row`. `load` imports
the one module that defines a type. The decorator itself does nothing, so importing a module that
registers types costs nothing.

Standard library only: `scripts/studio_surface.py` loads this file by path, without torch.
"""

import ast
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from functools import cached_property
from importlib import import_module
from pathlib import Path
from typing import Any, NamedTuple

__all__ = [
    "ARCH_REGISTRY",
    "DATASET_REGISTRY",
    "LOSS_REGISTRY",
    "METRIC_REGISTRY",
    "MODEL_REGISTRY",
    "Registry",
    "Row",
]


class Row(NamedTuple):
    module: str  # file stem under the registry's package
    scale: str | None  # constructor kwarg that takes the run scale
    meta: dict[str, Any]  # the decorator's other keywords


class Registry(Mapping[str, Row]):
    """`marker` is the name the registry is bound to: the scan looks for `@<marker>.register(`.

    The one accepted form: the decorator at column 0, directly above a module-level `class` or
    `def` (other decorators go above it), with keyword arguments that are literals. A module of
    the package that names the marker must use it for its import and these decorators and nothing
    else, so every other spelling (an alias, a call, an indented or commented-out decorator)
    fails the scan by file name instead of being skipped. `__init__.py` is not scanned.
    """

    def __init__(
        self,
        marker: str,
        package: str,
        *,
        required: Iterable[str] = (),
        optional: Iterable[str] = (),
        path: Path | None = None,
    ) -> None:
        self.marker, self.package = marker, package
        self.required = frozenset(required)
        self.allowed = self.required | frozenset(optional)
        self.path = path or Path(__file__).parents[2].joinpath(*package.split("."))

    def register[T](self, **meta: Any) -> Callable[[T], T]:  # noqa: ARG002
        """Declare the definition below as a type of this registry; the scan reads `meta`."""
        return lambda obj: obj

    def load(self, name: str) -> Any:
        """The registered class or function; imports its module and nothing else."""
        return getattr(import_module(f"{self.package}.{self[name].module}"), name)

    @cached_property
    def rows(self) -> dict[str, Row]:
        marker = re.escape(self.marker)
        declared = re.compile(
            rf"^@{marker}\.register\((.*?)\)[ \t]*(?:#[^\n]*)?\n(?:class|def) (\w+)",
            re.MULTILINE | re.DOTALL,
        )
        used = re.compile(rf"\b{marker}\b")
        rows: dict[str, Row] = {}
        files = sorted(self.path.glob("*.py"))
        if not files:
            msg = f"{self.path}: no source files to scan for {self.marker}"
            raise RuntimeError(msg)
        for file in files:
            text = file.read_text("utf-8-sig")
            if self.marker not in text or file.stem == "__init__":
                continue
            found = declared.findall(text)
            if not found or len(used.findall(text)) != len(found) + 1:
                msg = (
                    f"{file}: {self.marker} may appear only in its import and in "
                    f"@{self.marker}.register(...) at column 0, directly above a module-level "
                    f"class or def; {len(found)} such decorator(s) are readable"
                )
                raise RuntimeError(msg)
            for args, name in found:
                meta = self._keywords(file, name, args)
                if name in rows:
                    msg = f"{file}: {name} is already registered in {rows[name].module}.py"
                    raise RuntimeError(msg)
                rows[name] = Row(file.stem, meta.pop("scale", None), meta)
        return rows

    def _keywords(self, file: Path, name: str, args: str) -> dict[str, Any]:
        try:
            call: Any = ast.parse(f"f({args}\n)", mode="eval").body
            meta = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
            ok = (
                not call.args
                and len(meta) == len(call.keywords)  # no repeated keyword
                and self.required <= meta.keys() <= self.allowed
                and isinstance(meta.get("scale"), str | None)
            )
        except (SyntaxError, ValueError, AttributeError):
            ok = False
        if not ok:
            msg = (
                f"{file}: {name}: register() takes keyword literals; it needs "
                f"{sorted(self.required)} and allows {sorted(self.allowed)} "
                "(scale: the constructor's kwarg name or None)"
            )
            raise RuntimeError(msg)
        return meta

    def __getitem__(self, name: str) -> Row:
        return self.rows[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)


ARCH_REGISTRY = Registry("ARCH_REGISTRY", "neosr.archs", required=["scale"])
LOSS_REGISTRY = Registry("LOSS_REGISTRY", "neosr.losses")
METRIC_REGISTRY = Registry("METRIC_REGISTRY", "neosr.metrics", optional=["choices"])
DATASET_REGISTRY = Registry("DATASET_REGISTRY", "neosr.data")
MODEL_REGISTRY = Registry("MODEL_REGISTRY", "neosr.models")
