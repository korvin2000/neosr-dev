"""Loss contract (neosr/losses/base.py) and the objective builder (build_losses), design T1 §4.1-4.2."""

import ast
import inspect
import math
import tomllib
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import neosr.losses as losses
from neosr.config import ConfigError, load_config
from neosr.losses import LOSSES, SLOT_KEY, build_loss, build_losses
from neosr.losses.base import Context, Loss
from neosr.utils.factory import inject_scale, resolve

ROOT = Path(__file__).resolve().parents[1]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CPU = torch.device("cpu")
# today's `if train_opt.get("<slot>_opt")` chain (image.py:179-273)
OLD_SLOTS = ["pixel", "mssim", "fdl", "ncc", "kl", "consistency"]
OLD_SLOTS += ["msswd", "perceptual", "dists", "gan", "ldl", "ff"]
TRAIN_CONFIGS = sorted(
    p
    for p in (ROOT / "options").rglob("*.toml")
    if "train" in tomllib.loads(p.read_text("utf-8"))
)


class L1(Loss):
    def __init__(self, **kw) -> None:
        super().__init__()
        self.kw = kw
        self.w = nn.Parameter(
            torch.ones(())
        )  # a ctor-made parameter the builder must freeze

    def forward(self, x, y):
        return (x - y).abs().mean() * self.w


class MatMul(Loss):
    def forward(self, x, y):  # autocast-sensitive: bf16 under autocast unless fp32
        return torch.mm(x.flatten(1), y.flatten(1).T).mean()


class GanG(Loss):
    loss_weight = 0.1

    def __init__(self, **kw) -> None:
        super().__init__()
        self.kw = kw

    def evaluate(self, ctx):
        return ctx.sr.mean()

    def expand(self):
        return [self, GanD()]


class GanD(Loss):
    phase = "d"

    def evaluate(self, ctx):
        return ctx.gt.mean()


def fake_build(cfg, *, scale, section):  # noqa: ARG001
    kw = dict(cfg)
    return {"gan_loss": GanG}.get(kw.pop("type"), L1)(**kw)


@pytest.fixture
def stubs(monkeypatch):
    monkeypatch.setattr(losses, "build_loss", fake_build)


def ctx(scale: int = 2, dev: str = DEV, **kw) -> Context:
    g = torch.Generator(device=dev).manual_seed(0)
    lq = torch.rand(2, 3, 17, 23, device=dev, generator=g)
    sr = torch.rand(2, 3, 17 * scale, 23 * scale, device=dev, generator=g)
    gt = torch.rand(sr.shape, device=dev, generator=g)
    return Context(sr=sr.requires_grad_(), gt=gt, lq=lq, it=1, **kw)


def build(train, *, has_d=False):
    return build_losses(train, scale=2, device=CPU, has_d=has_d)


# ---- contract ---------------------------------------------------------------------------------


@pytest.mark.parametrize("dev", sorted({"cpu", DEV}))
@pytest.mark.parametrize("scale", [2, 3, 4])
def test_lq_up_equals_old_colour_target_bitwise(scale, dev):
    c = ctx(scale, dev)
    old = torch.clamp(
        F.interpolate(c.lq, scale_factor=scale, mode="bicubic", antialias=True),
        1 / 255,
        1,
    )  # image.py:525-534
    assert torch.equal(c.lq_up, old)
    assert c.lq_up is c.lq_up
    assert not c.lq_up.requires_grad


@pytest.mark.parametrize("dev", sorted({"cpu", DEV}))
def test_lq_up_is_fp32_under_autocast(dev):
    c = ctx(dev=dev)
    with torch.autocast(dev, dtype=torch.bfloat16):
        assert c.lq_up.dtype is torch.float32


def test_compute_applies_weight_once_and_logs():
    c, loss = ctx(), L1()
    loss.loss_weight, loss.log_key = 3.0, "l_g_pix"
    value = loss.compute(c)
    assert torch.equal(value, 3.0 * loss(c.sr, c.gt))
    assert torch.equal(c.logs["l_g_pix"], value.detach())
    assert not c.logs["l_g_pix"].requires_grad
    value.backward()
    assert c.sr.grad is not None
    assert c.sr.grad.abs().sum() > 0


def test_compute_without_log_key_logs_nothing():
    c = ctx()
    L1().compute(c)
    assert c.logs == {}


def test_target_lq_up():
    c, loss = ctx(), L1()
    loss.target = "lq_up"
    assert torch.equal(loss.compute(c), loss(c.sr, c.lq_up))


def test_non_scalar_raises():
    class PerPixel(Loss):
        def forward(self, x, y):
            return (x - y).abs()

    with pytest.raises(ValueError, match=r"PerPixel returned shape \(2, 3, 34, 46\)"):
        PerPixel().compute(ctx())


@pytest.mark.parametrize("dev", sorted({"cpu", DEV}))
def test_fp32_disables_autocast(dev):
    c = ctx(dev=dev)
    with torch.autocast(dev, dtype=torch.bfloat16):
        assert MatMul().compute(c).dtype is torch.bfloat16
        fp32 = MatMul()
        fp32.fp32 = True
        c.sr = c.sr.detach().bfloat16()  # a bf16 generator output is cast up too
        assert fp32.compute(c).dtype is torch.float32


# ---- build_loss (T3 L-D) ----------------------------------------------------------------------


def test_build_loss_errors():
    with pytest.raises(
        ConfigError, match=r"\[train.pixel_opt\] unknown type 'L1loss'.*'L1Loss'"
    ):
        build_loss({"type": "L1loss"}, section="train.pixel_opt")
    with pytest.raises(ConfigError, match=r"\[train.loss\] L1Loss: .*reduce"):
        build_loss({"type": "L1Loss", "reduce": "mean"})


def test_losses_table_is_complete_and_scale_free():
    registered = {
        node.name
        for f in (ROOT / "neosr" / "losses").glob("*.py")
        for node in ast.walk(ast.parse(f.read_text("utf-8")))
        if isinstance(node, ast.ClassDef)
        and any("LOSS_REGISTRY" in ast.unparse(d) for d in node.decorator_list)
    }
    if registered:  # until T3-Z deletes the decorators
        assert registered == set(LOSSES)
    # PatchLoss, the only loss that took the run scale, was removed (Phase L1)
    for name, (_, scale_kw) in LOSSES.items():
        obj = resolve(LOSSES, "neosr.losses", "t", name)
        assert scale_kw is None, name
        assert not {"scale", "upscale"} & set(inspect.signature(obj).parameters), name


# ---- build_losses -----------------------------------------------------------------------------


def test_weights_targets_and_freezing(stubs):  # noqa: ARG001
    train = {
        "optim_g": {"type": "adan_sf"},  # not a loss slot
        "ema": 0.999,
        "pixel_opt": {"type": "L1Loss", "loss_weight": 2},
        "perceptual_opt": {"type": "L1Loss", "perceptual_weight": 0.5},
        "edge_opt": {"type": "L1Loss", "target": "lq"},
        "empty_opt": {},
        "gan_opt": {"type": "gan_loss", "gan_type": "bce"},
    }
    g, d = build(train, has_d=True)
    assert [(t.log_key, t.loss_weight, t.target) for t in g] == [
        ("l_g_pix", 2.0, "gt"),
        ("l_g_percep", 0.5, "gt"),
        ("l_g_edge", 1.0, "lq_up"),
        ("l_g_gan", 0.1, "gt"),  # class default
    ]
    assert [type(t) for t in d] == [GanD]
    assert g[-1].kw == {"gan_type": "bce"}  # weight/target never reach the ctor
    assert all(not p.requires_grad for t in g for p in t.parameters())
    assert not any(t.training for t in g + d)


def test_legacy_globals(stubs):  # noqa: ARG001
    train = {
        "match_lq_colors": True,
        "wavelet_guided": True,
        "wavelet_init": 80000,
        "consistency_opt": {"type": "consistency_loss"},
        "msswd_opt": {"type": "msswd_loss", "target": "gt"},  # an explicit target wins
        "mssim_opt": {"type": "mssim_loss"},
        "gan_opt": {"type": "gan_loss"},
    }
    g, _ = build(train, has_d=True)
    assert {t.log_key: t.target for t in g} == {
        "l_g_consistency": "lq_up",
        "l_g_msswd": "gt",
        "l_g_mssim": "gt",
        "l_g_gan": "gt",
    }
    assert g[-1].kw == {"wavelet_init": 80000}


def test_zero_weight_pruning_keeps_the_d_term(stubs):  # noqa: ARG001
    train = {
        "pixel_opt": {"type": "L1Loss"},
        "ldl_opt": {"type": "L1Loss", "loss_weight": 0},
        "gan_opt": {"type": "gan_loss", "loss_weight": 0.0},
    }
    g, d = build(train, has_d=True)
    assert [t.log_key for t in g] == ["l_g_pix"]
    assert [type(t) for t in d] == [GanD]


GAN = {"type": "gan_loss"}
PIX = {"type": "L1Loss"}


@pytest.mark.parametrize(
    ("train", "has_d", "match"),
    [
        ({"pixel_opt": PIX}, True, "network_d requires gan_opt"),
        ({"pixel_opt": PIX, "gan_opt": GAN}, False, "gan_opt requires network_d"),
        (
            {"pixel_opt": PIX, "wavelet_guided": True},
            False,
            "wavelet_guided requires gan_opt",
        ),
        (
            {"pixel_opt": {**PIX, "loss_weight": 1, "perceptual_weight": 1}},
            False,
            "not both",
        ),
        ({"pixel_opt": {**PIX, "loss_weight": math.nan}}, False, "a finite number"),
        ({"pixel_opt": {**PIX, "loss_weight": math.inf}}, False, "a finite number"),
        ({"pixel_opt": {**PIX, "loss_weight": "oops"}}, False, "a finite number"),
        ({"pixel_opt": {**PIX, "loss_weight": []}}, False, "a finite number"),
        ({"pixel_opt": {**PIX, "loss_weight": True}}, False, "a finite number"),
        ({"perceptual_opt": {**PIX, "perceptual_weight": "1"}}, False, "a finite"),
        ({"pixel_opt": {**PIX, "target": "lr"}}, False, "target must be one of"),
        ({"pixel_opt": {**PIX, "target": []}}, False, "target must be one of"),
        ({"pixel_opt": PIX, "total_opt": PIX}, False, "reserved"),  # l_g_total
        ({"pixel_opt": PIX, "pix_opt": PIX}, False, "unique"),  # both log l_g_pix
        (
            {"pixel_opt": {**PIX, "loss_weight": 0}},
            False,
            "at least one generator loss",
        ),
        ({"gan_opt": {**GAN, "loss_weight": 0}}, True, "at least one generator loss"),
        ({}, False, "at least one generator loss"),
    ],
)
def test_config_errors(stubs, train, has_d, match):  # noqa: ARG001
    with pytest.raises(ConfigError, match=match):
        build(train, has_d=has_d)


# ---- dry run over every shipped training config -----------------------------------------------


def dry_build(cfg, *, scale, section):
    """Resolve the real class and bind the config kwargs to its ctor; build a stub (B1/A2 not landed).

    `wavelet_init` is the one kwarg the builder adds that only A2's gan_loss accepts.
    """
    kw = dict(cfg)
    name = kw.pop("type")
    obj = resolve(LOSSES, "neosr.losses", section, name)
    inject_scale(LOSSES, section, name, kw, scale)
    stub = GanG(**kw) if name == "gan_loss" else L1(**kw)
    if name == "gan_loss":
        kw.pop("wavelet_init", None)
    inspect.signature(obj).bind(**kw)
    return stub


@pytest.mark.parametrize(
    "path", TRAIN_CONFIGS, ids=lambda p: str(p.relative_to(ROOT / "options"))
)
def test_shipped_configs_dry_run(path, monkeypatch):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = load_config(path, is_train=True, root=ROOT)
    monkeypatch.setattr(losses, "build_loss", dry_build)
    train = cfg["train"]
    g, d = build_losses(train, scale=cfg["scale"], device=CPU, has_d="network_d" in cfg)

    old = {f"l_g_{SLOT_KEY.get(s, s)}" for s in OLD_SLOTS if train.get(f"{s}_opt")}
    assert {t.log_key for t in g} == old
    assert bool(d) == bool(train.get("gan_opt"))
    for t in g:
        lq = t.log_key in {"l_g_consistency", "l_g_msswd"} and train.get(
            "match_lq_colors"
        )
        assert t.target == ("lq_up" if lq else "gt")
        if t.log_key == "l_g_gan" and train.get("wavelet_guided"):
            assert t.kw["wavelet_init"] == train.get("wavelet_init", 0)


# ---- T1 §6 A1 gate: the real constructors (runs once B1/A2 make every class a Loss) ----------

MIGRATED = all(
    issubclass(resolve(LOSSES, "neosr.losses", "t", n), Loss) for n in LOSSES
)


@pytest.mark.xfail(
    not MIGRATED, reason="B1/A2: classes not yet Loss subclasses", strict=True
)
def test_shipped_configs_real_constructors():
    """G terms: finite scalar, non-zero sr.grad. D terms: finite scalar, sr untouched.
    Every loss parameter and buffer is frozen and on the run device."""
    dev = torch.device(DEV)
    net_d = nn.Conv2d(3, 1, 3, padding=1).to(dev)  # tiny stand-in critic
    for path in TRAIN_CONFIGS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = load_config(path, is_train=True, root=ROOT)
        train, s = cfg["train"], cfg["scale"]
        g, d = build_losses(train, scale=s, device=dev, has_d="network_d" in cfg)
        for t in g + d:
            where = (path.name, t.phase, t.log_key)
            for p in [*t.parameters(), *t.buffers()]:
                assert p.device.type == dev.type, where
                assert not p.requires_grad, where
            c = ctx(s, DEV, net_d=net_d, net_d_adv=net_d)
            value = t.compute(c)
            assert value.ndim == 0, where
            assert torch.isfinite(value), where
            if t.phase == "g":
                value.backward()
                assert c.sr.grad is not None, where
                assert c.sr.grad.abs().sum() > 0, where
            elif value.requires_grad:
                value.backward()
                assert c.sr.grad is None, where  # D term must see a detached sr
