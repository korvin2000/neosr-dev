"""The lazy builder tables and the shared factory. Tables are checked as their packages adopt them."""

import ast
import importlib
import inspect
import textwrap
from typing import Any

import pytest
from torch import nn

from neosr.config import ConfigError
from neosr.utils.factory import construct, inject_scale, resolve

# package -> name of its table; a package without the table is skipped until its lane lands
TABLES = {
    "neosr.archs": "ARCHS",
    "neosr.losses": "LOSSES",
    "neosr.data": "DATASETS",
    "neosr.models": "MODELS",
    "neosr.metrics": "METRICS",
}
# networks that take no run scale (discriminators, feature extractors)
NON_SR_ARCHS = {"dunet", "ea2fpn", "metagan", "patchgan", "resgan", "unet"}
# reviewed exceptions: the scale kwarg is read from **kwargs, so the signature cannot show it
SCALE_VIA_KWARGS = {("neosr.archs", "omnisr")}


def table_rows() -> list[tuple[str, str, tuple[str, str | None]]]:
    rows = []
    for package, attr in TABLES.items():
        table = getattr(importlib.import_module(package), attr, None)
        rows += [(package, name, row) for name, row in (table or {}).items()]
    return rows


ROWS = table_rows()


@pytest.mark.skipif(not ROWS, reason="no package table yet")
@pytest.mark.parametrize(
    ("package", "name", "row"), ROWS, ids=[f"{p}:{n}" for p, n, _ in ROWS]
)
def test_row_resolves_and_scale_kw_exists(
    package: str, name: str, row: tuple[str, str | None]
) -> None:
    obj = resolve({name: row}, package, "test", name)
    assert callable(obj)
    scale_kw = row[1]
    if scale_kw is not None and (package, name) not in SCALE_VIA_KWARGS:
        assert scale_reaches(obj, scale_kw), scale_kw


def scale_reaches(obj: Any, kw: str) -> bool:
    """`kw` is a named parameter, or a factory forwards its **kwargs without setting `kw` itself.

    The forwarded-to constructor is checked by its own table row.
    """
    target = obj.__init__ if inspect.isclass(obj) else obj
    params = inspect.signature(target).parameters
    if kw in params:
        return True
    var_kw = next((p.name for p in params.values() if p.kind is p.VAR_KEYWORD), None)
    if var_kw is None or inspect.isclass(obj):
        return False
    calls = [
        n
        for n in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(obj))))
        if isinstance(n, ast.Call)
    ]
    forwards = any(
        k.arg is None and isinstance(k.value, ast.Name) and k.value.id == var_kw
        for c in calls
        for k in c.keywords
    )
    return forwards and not any(k.arg == kw for c in calls for k in c.keywords)


def test_scale_free_archs_are_the_non_sr_networks() -> None:
    archs = importlib.import_module("neosr.archs")
    if not hasattr(archs, "ARCHS"):
        pytest.skip("neosr.archs has no ARCHS table yet")
    scale_free = {name for name, (_, kw) in archs.ARCHS.items() if kw is None}
    assert scale_free - {"VGGFeatureExtractor"} == NON_SR_ARCHS


class Net(nn.Module):
    def __init__(self, channels: int = 3, upsampling: int = 4) -> None:
        super().__init__()
        self.channels, self.upsampling = channels, upsampling


class Broken(nn.Module):
    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        len(channels)  # a bug inside the ctor: TypeError that is not about the kwargs


FAKE = {"Net": ("unused", "upsampling"), "plain": ("unused", None)}


def forwards(**kwargs):
    return Net(channels=3, **kwargs)


def drops(**kwargs):  # noqa: ARG001
    return Net(channels=3)


def hard_codes(**kwargs):
    return Net(upsampling=2, **kwargs)


class Sink(nn.Module):
    def __init__(self, **kwargs) -> None:  # noqa: ARG002
        super().__init__()


def test_scale_reaches_rejects_kwargs_sinks() -> None:
    assert scale_reaches(Net, "upsampling")
    assert scale_reaches(forwards, "upsampling")
    assert not scale_reaches(drops, "upsampling")
    assert not scale_reaches(hard_codes, "upsampling")
    assert not scale_reaches(Sink, "upsampling")


def test_resolve_unknown_type_suggests() -> None:
    with pytest.raises(
        ConfigError, match=r"\[network_g\] unknown type 'Nett' \(did you mean 'Net'\?\)"
    ):
        resolve(FAKE, "neosr.archs", "network_g", "Nett")
    with pytest.raises(ConfigError, match="unknown type None"):
        resolve(FAKE, "neosr.archs", "network_g", None)
    for bad in ([], {}):  # an opaque section lets any TOML value through as `type`
        with pytest.raises(ConfigError, match="unknown type"):
            resolve(FAKE, "neosr.archs", "network_g", bad)


def test_resolve_imports_lazily() -> None:
    assert (
        resolve({"ConfigError": ("config", None)}, "neosr", "x", "ConfigError")
        is ConfigError
    )


def test_unknown_model_type() -> None:
    models = {"image": ("image", None), "otf": ("otf", None)}
    with pytest.raises(ConfigError, match="did you mean 'image'"):
        resolve(models, "neosr.models", "model_type", "imagee")


def test_inject_scale() -> None:
    kwargs: dict = {}
    inject_scale(FAKE, "network_g", "Net", kwargs, 2)
    assert kwargs == {"upsampling": 2}
    inject_scale(
        FAKE, "network_g", "Net", kwargs, 2
    )  # an explicit matching value is fine
    with pytest.raises(ConfigError, match="upsampling=4 contradicts scale=2"):
        inject_scale(FAKE, "network_g", "Net", {"upsampling": 4}, 2)
    with pytest.raises(ConfigError, match="needs the run scale"):
        inject_scale(FAKE, "network_g", "Net", {}, None)
    plain: dict = {}
    inject_scale(FAKE, "network_d", "plain", plain, None)
    assert plain == {}


def test_construct_maps_only_kwarg_errors_to_config_error() -> None:
    assert construct(Net, "network_g", "Net", {"channels": 4}).channels == 4
    with pytest.raises(ConfigError, match=r"\[network_g\] Net: .*chanels"):
        construct(Net, "network_g", "Net", {"chanels": 4})
    with pytest.raises(
        TypeError, match="has no len"
    ):  # ConfigError is a ValueError, so this is the original
        construct(Broken, "network_g", "Broken", {})
