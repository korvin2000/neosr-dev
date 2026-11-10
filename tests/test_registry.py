"""The registries (design T7) and the shared factory: the scan, the rows of the five packages, construction."""

import ast
import inspect
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from torch import nn

from neosr.config import ConfigError
from neosr.utils import registry as registry_module
from neosr.utils.factory import build, construct, inject_scale, resolve
from neosr.utils.registry import (
    ARCH_REGISTRY,
    DATASET_REGISTRY,
    LOSS_REGISTRY,
    METRIC_REGISTRY,
    MODEL_REGISTRY,
    Registry,
    Row,
)

REGISTRIES = (
    ARCH_REGISTRY,
    LOSS_REGISTRY,
    METRIC_REGISTRY,
    DATASET_REGISTRY,
    MODEL_REGISTRY,
)
# reviewed exceptions: the scale kwarg is read from **kwargs, so the signature cannot show it
SCALE_VIA_KWARGS = {("neosr.archs", "omnisr")}
# reviewed: a scale-free network's parameter that only shares its name with a scale kwarg
NOT_THE_RUN_SCALE = {"resgan": {"upscale"}}  # a bool switch of the U-Net discriminator
ROWS = [(reg, name) for reg in REGISTRIES for name in reg]


# ---------------------------------------------------------------- the rows of the five packages
def test_every_registry_has_rows() -> None:
    assert all(len(reg) for reg in REGISTRIES)


@pytest.mark.parametrize(
    ("reg", "name"), ROWS, ids=[f"{reg.package}:{name}" for reg, name in ROWS]
)
def test_row_resolves_and_scale_kw_exists(reg: Registry, name: str) -> None:
    obj = resolve(reg, "test", name)
    assert callable(obj)
    assert obj.__name__ == name  # the type name is the Python name
    scale_kw = reg[name].scale
    if scale_kw is not None and (reg.package, name) not in SCALE_VIA_KWARGS:
        assert scale_reaches(obj, scale_kw), scale_kw


def scale_reaches(obj: Any, kw: str) -> bool:
    """`kw` is a named parameter, or a factory forwards its **kwargs without setting `kw` itself.

    The forwarded-to constructor is checked by its own row.
    """
    target = obj.__init__ if inspect.isclass(obj) else obj
    params = inspect.signature(target).parameters
    if kw in params:
        return True
    var_kw = next((p.name for p in params.values() if p.kind is p.VAR_KEYWORD), None)
    if var_kw is None or inspect.isclass(obj):
        return False
    body = ast.parse(textwrap.dedent(inspect.getsource(obj))).body[0].body  # type: ignore[attr-defined]
    calls = [n for stmt in body for n in ast.walk(stmt) if isinstance(n, ast.Call)]
    forwards = any(
        k.arg is None and isinstance(k.value, ast.Name) and k.value.id == var_kw
        for c in calls
        for k in c.keywords
    )
    return forwards and not any(k.arg == kw for c in calls for k in c.keywords)


def test_registry_module_is_stdlib_only() -> None:
    """`scripts/studio_surface.py` loads it without torch; arch modules import it for free."""
    tree = ast.parse(Path(registry_module.__file__).read_text("utf-8"))
    roots = {
        (node.module if isinstance(node, ast.ImportFrom) else alias.name).split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    }
    assert roots <= sys.stdlib_module_names, roots


def bound(node: ast.stmt) -> list[str]:
    if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
        return [node.name]
    targets = getattr(node, "targets", [getattr(node, "target", None)])
    return [t.id for t in targets if isinstance(t, ast.Name)]


@pytest.mark.parametrize("reg", REGISTRIES, ids=[reg.marker for reg in REGISTRIES])
def test_scan_equals_an_independent_ast_reading(reg: Registry) -> None:
    """The scan is a regex over a restricted form; `ast` is the oracle for what the files declare.

    Also: a registered name is bound once in its module, so `load` returns the decorated object.
    """
    rows = {}
    for file in sorted(reg.path.glob("*.py")):
        tree = ast.parse(file.read_text("utf-8"))
        names = [name for node in tree.body for name in bound(node)]
        for node in ast.walk(tree):
            for deco in getattr(node, "decorator_list", []):
                if reg.marker in ast.unparse(deco):
                    assert node in tree.body, f"{file.name}: {node.name} is nested"
                    assert names.count(node.name) == 1, f"{file.name}: {node.name}"
                    kw = {k.arg: ast.literal_eval(k.value) for k in deco.keywords}
                    rows[node.name] = Row(file.stem, kw.pop("scale", None), kw)
    assert rows == dict(reg)


def reachable_params(obj: Any) -> set[str]:
    """Parameters of `obj`, and for a factory those of the module-level classes it calls."""
    if inspect.isclass(obj):
        return set(inspect.signature(obj.__init__).parameters)
    params = set(inspect.signature(obj).parameters)
    module = sys.modules[obj.__module__]
    for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(obj)))):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            callee = getattr(module, node.func.id, None)
            if inspect.isclass(callee):
                params |= set(inspect.signature(callee.__init__).parameters)
    return params


def test_scale_free_archs_have_no_scale_kwarg() -> None:
    """`scale=None` on a network that does take a scale would silently build it at its default."""
    scale_kws = {row.scale for row in ARCH_REGISTRY.values()} - {None}
    wrong = {
        name: scale_kws & reachable_params(ARCH_REGISTRY.load(name))
        for name, row in ARCH_REGISTRY.items()
        if row.scale is None
    }
    assert {name: kws for name, kws in wrong.items() if kws} == NOT_THE_RUN_SCALE


# ---------------------------------------------------------------- the scan, on a throwaway package
INIT = "from neosr.utils.registry import Registry\nFAKE = Registry('FAKE', 'fakepkg')\n"
HEAD = "from fakepkg import FAKE\n"
NETS = """
from torch import nn


@FAKE.register(scale="upsampling")
class Net(nn.Module):
    def __init__(self, channels: int = 3, upsampling: int = 4) -> None:
        super().__init__()
        self.channels, self.upsampling = channels, upsampling


@FAKE.register(
    scale="upsampling",
    note=("a", ["b"]),  # nested literals and comments are fine
)  # so is a trailing comment
def forwards(**kwargs):
    return Net(channels=3, **kwargs)


@staticmethod
@FAKE.register(scale=None)
def echo(opt: dict) -> dict:
    return opt


class Helper:  # an undecorated definition is not a row
    pass
"""
OTHER = "@FAKE.register(scale=None)\nclass Other:\n    pass\n"


def fake(tmp_path: Path, monkeypatch, **sources: str) -> Registry:
    """A registry over `tmp_path/fakepkg`, importable; `sources` are module name -> body."""
    pkg = tmp_path / "fakepkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(INIT, "utf-8")
    for stem, body in sources.items():
        (pkg / f"{stem}.py").write_text(HEAD + body, "utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [m for m in sys.modules if m.split(".")[0] == "fakepkg"]:
        monkeypatch.delitem(sys.modules, name)
    return Registry("FAKE", "fakepkg", required=["scale"], optional=["note"], path=pkg)


def test_scan_reads_the_decorators_and_imports_nothing(tmp_path, monkeypatch) -> None:
    reg = fake(tmp_path, monkeypatch, nets=NETS, other=OTHER)
    assert dict(reg) == {
        "Net": Row("nets", "upsampling", {}),
        "forwards": Row("nets", "upsampling", {"note": ("a", ["b"])}),
        "echo": Row("nets", None, {}),
        "Other": Row("other", None, {}),
    }
    assert not [m for m in sys.modules if m.startswith("fakepkg")]
    net = reg.load("forwards")(upsampling=2)
    assert (type(net).__name__, net.upsampling) == ("Net", 2)
    assert "fakepkg.nets" in sys.modules
    assert "fakepkg.other" not in sys.modules  # one lookup, one module


def test_register_returns_the_object_unchanged() -> None:
    assert ARCH_REGISTRY.register(scale=None)(Row) is Row


@pytest.mark.parametrize(
    ("body", "error"),
    [
        ("if True:\n    @FAKE.register(scale=None)\n    class A: pass\n", "readable"),
        ("@FAKE.register(scale=None)\n@staticmethod\ndef a(): pass\n", "readable"),
        ("@FAKE.register(scale=None)\n\nclass A: pass\n", "readable"),
        ("class A: pass\nFAKE.register(scale=None)(A)\n", "readable"),
        ("@FAKE.register(scale=None)\nasync def a(): pass\n", "readable"),
        ("KW = None\n@FAKE.register(scale=KW)\nclass A: pass\n", "keyword literals"),
        ("@FAKE.register('upscale')\nclass A: pass\n", "keyword literals"),
        ("@FAKE.register()\nclass A: pass\n", "keyword literals"),
        ("@FAKE.register(scale=None, scal='x')\nclass A: pass\n", "keyword literals"),
        ("@FAKE.register(scale=None, scale=None)\nclass A: pass\n", "keyword literals"),
        ("@FAKE.register(**{'scale': None})\nclass A: pass\n", "keyword literals"),
        ("@FAKE.register(scale=[])\nclass A: pass\n", "keyword literals"),
        ("@FAKE.register (scale=None)\nclass A: pass\n", "readable"),
        ("REG = FAKE\n@REG.register(scale=None)\nclass A: pass\n", "readable"),
        ("# @FAKE.register(scale=None)\nclass A: pass\n", "readable"),
        (OTHER + OTHER, "already registered"),
    ],
)
def test_scan_rejects_what_it_cannot_read(tmp_path, monkeypatch, body, error) -> None:
    reg = fake(tmp_path, monkeypatch, bad=body)
    with pytest.raises(RuntimeError, match=rf"bad\.py.*{error}"):
        len(reg)


def test_a_type_registered_in_two_modules_fails(tmp_path, monkeypatch) -> None:
    reg = fake(tmp_path, monkeypatch, a=OTHER, b=OTHER)
    with pytest.raises(
        RuntimeError, match=r"b\.py: Other is already registered in a\.py"
    ):
        len(reg)


# ---------------------------------------------------------------- the factory
class Broken(nn.Module):
    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        len(channels)  # a bug inside the ctor: TypeError that is not about the kwargs


class Sink(nn.Module):
    def __init__(self, **kwargs) -> None:  # noqa: ARG002
        super().__init__()


def test_scale_reaches_rejects_kwargs_sinks(tmp_path, monkeypatch) -> None:
    body = NETS + (
        "def drops(**kwargs):\n    return Net(channels=3)\n\n"
        "def hard_codes(**kwargs):\n    return Net(upsampling=2, **kwargs)\n"
    )
    reg = fake(tmp_path, monkeypatch, nets=body)
    nets = sys.modules[reg.load("Net").__module__]
    assert scale_reaches(nets.Net, "upsampling")
    assert scale_reaches(nets.forwards, "upsampling")
    assert not scale_reaches(nets.drops, "upsampling")
    assert not scale_reaches(nets.hard_codes, "upsampling")
    assert not scale_reaches(Sink, "upsampling")


def test_resolve_unknown_type_suggests(tmp_path, monkeypatch) -> None:
    reg = fake(tmp_path, monkeypatch, nets=NETS)
    with pytest.raises(
        ConfigError, match=r"\[network_g\] unknown type 'Nett' \(did you mean 'Net'\?\)"
    ):
        resolve(reg, "network_g", "Nett")
    with pytest.raises(ConfigError, match="unknown type None"):
        resolve(reg, "network_g", None)
    for bad in ([], {}):  # an opaque section lets any TOML value through as `type`
        with pytest.raises(ConfigError, match="unknown type"):
            resolve(reg, "network_g", bad)


def test_unknown_model_type() -> None:
    with pytest.raises(ConfigError, match="did you mean 'image'"):
        resolve(MODEL_REGISTRY, "model_type", "imagee")


def test_inject_scale(tmp_path, monkeypatch) -> None:
    reg = fake(tmp_path, monkeypatch, nets=NETS)
    kwargs: dict = {}
    inject_scale(reg, "network_g", "Net", kwargs, 2)
    assert kwargs == {"upsampling": 2}
    inject_scale(
        reg, "network_g", "Net", kwargs, 2
    )  # an explicit matching value is fine
    with pytest.raises(ConfigError, match="upsampling=4 contradicts scale=2"):
        inject_scale(reg, "network_g", "Net", {"upsampling": 4}, 2)
    with pytest.raises(ConfigError, match="needs the run scale"):
        inject_scale(reg, "network_g", "Net", {}, None)
    plain: dict = {}
    inject_scale(reg, "network_d", "echo", plain, None)
    assert plain == {}


def test_construct_maps_only_kwarg_errors_to_config_error(
    tmp_path, monkeypatch
) -> None:
    net = fake(tmp_path, monkeypatch, nets=NETS).load("Net")
    assert construct(net, "network_g", "Net", {"channels": 4}).channels == 4
    with pytest.raises(ConfigError, match=r"\[network_g\] Net: .*chanels"):
        construct(net, "network_g", "Net", {"chanels": 4})
    with pytest.raises(
        TypeError, match="has no len"
    ):  # ConfigError is a ValueError, so this is the original
        construct(Broken, "network_g", "Broken", {})


def test_build_copies_cfg_and_dispatches_both_call_shapes(
    tmp_path, monkeypatch
) -> None:
    reg = fake(tmp_path, monkeypatch, nets=NETS)
    cfg = {"type": "Net", "channels": 4}
    net = build(reg, "network_g", cfg, scale=2)
    assert (net.channels, net.upsampling) == (4, 2)
    assert cfg == {"type": "Net", "channels": 4}  # the caller's table is untouched
    nested = {"type": "echo", "phase": "train", "aug": [1]}
    opt = build(reg, "datasets.train", nested, whole=True)
    assert opt == nested  # one positional dict that keeps `type`
    assert opt["aug"] is not nested["aug"]  # a deep copy
    with pytest.raises(ConfigError, match="needs the run scale"):
        build(reg, "network_g", {"type": "Net"})
    with pytest.raises(ConfigError, match=r"\[network_g\] Net: .*chanels"):
        build(reg, "network_g", {"type": "Net", "chanels": 4}, scale=2)
    with pytest.raises(ConfigError, match="unknown type None"):
        build(reg, "network_g", {})
