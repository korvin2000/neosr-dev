"""Shared lookup and construction for the per-package builders (archs, losses, data, models, metrics)."""

import importlib
import inspect
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from neosr.config import ConfigError

Table = Mapping[
    str, tuple[str, str | None]
]  # name -> (module under the package, scale kwarg | None)


def resolve(table: Table, package: str, section: str, name: Any) -> Any:
    if not isinstance(name, str) or name not in table:
        msg = f"[{section}] unknown type {name!r}"
        raise ConfigError(msg, key=str(name), known=table)
    return getattr(importlib.import_module(f"{package}.{table[name][0]}"), name)


def inject_scale(
    table: Table, section: str, name: str, kwargs: dict[str, Any], scale: int | None
) -> None:
    """Pass the run scale under the target's native kwarg name; an explicit contradicting value is an error."""
    kw = table[name][1]
    if kw is None:
        return
    if scale is None:
        msg = f"[{section}] {name} needs the run scale"
        raise ConfigError(msg)
    if kwargs.setdefault(kw, scale) != scale:
        msg = f"[{section}] {kw}={kwargs[kw]!r} contradicts scale={scale}"
        raise ConfigError(msg)


def construct(
    obj: Any, section: str, name: str, kwargs: dict[str, Any], **extra: Any
) -> Any:
    try:
        return obj(**kwargs, **extra)
    except TypeError as err:
        try:
            inspect.signature(obj).bind(**kwargs, **extra)
        except TypeError as bad:  # the kwargs themselves are wrong: a config problem
            msg = f"[{section}] {name}: {bad}"
            raise ConfigError(msg) from err
        raise  # a bug inside the constructor: keep the TypeError and its traceback


def build(
    table: Table,
    package: str,
    section: str,
    cfg: Mapping[str, Any],
    *,
    scale: int | None = None,
    whole: bool = False,
) -> Any:
    """Build `cfg["type"]` from a copy of `cfg` with the run scale injected: as keyword arguments
    without `type`, or (`whole`, the dataset shape) as one positional dict that keeps it."""
    kwargs = deepcopy(dict(cfg))
    name = kwargs.get("type") if whole else kwargs.pop("type", None)
    obj = resolve(table, package, section, name)
    inject_scale(table, section, name, kwargs, scale)
    return obj(kwargs) if whole else construct(obj, section, name, kwargs)
