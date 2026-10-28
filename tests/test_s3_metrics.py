"""Stage 3 A3: TOPIQ-FR `dataset` weights (kadid | pipal) and the LPIPS v0.1 metric. CPU; tests that
need pretrained weights run only when the torch hub cache holds them (no download)."""

from pathlib import Path

import pytest
import torch

from neosr.config import ConfigError
from neosr.metrics import build_metric, calculate
from neosr.utils.img_util import quantize

HUB = Path(torch.hub.get_dir()) / "checkpoints"
ALEX = HUB / "alexnet-owt-7be5be79.pth"
LPIPS_ALEX = HUB / "LPIPS_v0.1_alex-df73285e.pth"
KADID = HUB / "cfanet_fr_kadid_res50-2c4cc61d.pth"
CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def fresh_nets(monkeypatch):
    monkeypatch.setattr(calculate, "_metric_nets", {})


class _Fake(torch.nn.Module):
    def __init__(self, tag) -> None:
        super().__init__()
        self.tag, self.w = tag, torch.nn.Parameter(torch.ones(()))

    def forward(self, x, y):
        return (x - y).abs().mean() * self.w


@pytest.mark.parametrize(
    ("cfg", "match"),
    [
        (
            {"type": "calculate_topiq", "dataset": "kadidd"},
            r"dataset must be one of \['kadid', 'pipal'\]",
        ),
        (
            {"type": "calculate_lpips", "net": "squeeze"},
            r"net must be one of \['alex', 'vgg'\]",
        ),
    ],
)
def test_variant_values_are_checked_when_bound(cfg, match):
    with pytest.raises(ConfigError, match=match):
        build_metric(cfg, CPU)


@pytest.mark.parametrize(
    ("name", "key", "values"),
    [("topiq", "dataset", ("kadid", "pipal")), ("lpips", "net", ("alex", "vgg"))],
)
def test_one_cached_net_per_variant(monkeypatch, name, key, values):
    built = []
    monkeypatch.setitem(
        calculate._metric_builders, name, lambda v: built.append(v) or _Fake(v)
    )
    img = quantize(torch.rand(1, 3, 8, 8))
    for v in (*values, *values):
        build_metric({"type": f"calculate_{name}", key: v}, CPU)(img, img.flip(-1))
    assert built == list(values)
    assert {k: n.tag for k, n in calculate._metric_nets.items()} == {
        (name, v): v for v in values
    }


def test_lpips_is_lower_better_by_default(tmp_path):
    from tests.test_model import build, loader, make_opt  # noqa: PLC0415

    m = build(make_opt(tmp_path))
    m.opt["val"]["metrics"] = {"lpips": {"type": "calculate_lpips"}}
    m.metrics = {"lpips": lambda *_: torch.tensor(0.3)}
    m.validation(loader(), 1, None, save_img=False)
    assert m.best_metric_results["valset"]["lpips"]["better"] == "lower"


def test_topiq_pipal_weights_are_strict_and_named(monkeypatch):
    from neosr.metrics import topiq as tq  # noqa: PLC0415

    assert tq.WEIGHTS["pipal"][1].endswith("/cfanet_fr_pipal_res50-69bbe5ba.pth")
    if not KADID.is_file():
        pytest.skip("TOPIQ weights not cached")
    urls = []
    real = torch.load(KADID, map_location="cpu", weights_only=True)

    def hub(url, **_):
        urls.append(url)
        return {"params": {**real["params"], "extra": torch.zeros(1)}}

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", hub)
    # the hub branch, whatever local copy exists
    monkeypatch.setitem(tq.WEIGHTS, "pipal", ("absent.pth", tq.WEIGHTS["pipal"][1]))
    with pytest.raises(RuntimeError, match="Unexpected key.*extra"):
        tq.topiq(dataset="pipal")
    assert urls == [tq.WEIGHTS["pipal"][1]]


def _lin(**extra):
    chns = (64, 192, 384, 256, 256)
    return {
        f"lin{k}.model.1.weight": torch.rand(1, c, 1, 1) for k, c in enumerate(chns)
    } | extra


@pytest.mark.skipif(not ALEX.is_file(), reason="AlexNet weights not cached")
def test_lpips_loads_the_official_layout_strictly(monkeypatch):
    from neosr.metrics.lpips import lpips  # noqa: PLC0415

    sd = _lin()
    monkeypatch.setattr(
        torch.hub, "load_state_dict_from_url", lambda *_a, **_k: dict(sd)
    )
    net = lpips("alex")
    assert torch.equal(net.lins[2].weight, sd["lin2.model.1.weight"])
    assert set(net.state_dict()) == {f"lins.{k}.weight" for k in range(5)} | {
        k for k in net.state_dict() if k.startswith("slices.")
    }
    monkeypatch.setattr(
        torch.hub,
        "load_state_dict_from_url",
        lambda *_a, **_k: _lin(extra=torch.ones(1)),
    )
    with pytest.raises(RuntimeError, match="Unexpected key"):
        lpips("alex")


@pytest.mark.skipif(
    not (ALEX.is_file() and LPIPS_ALEX.is_file()), reason="LPIPS weights not cached"
)
def test_lpips_matches_the_reference_value():
    """0.10353731 is pyiqa's reference LPIPS (v0.1, alex) of this pair (probe in PROGRESS.md)."""
    g = torch.Generator().manual_seed(0)
    x, y = torch.rand(1, 3, 64, 64, generator=g), torch.rand(1, 3, 64, 64, generator=g)
    metric = build_metric({"type": "calculate_lpips"}, CPU)
    q = quantize(x), quantize(y)
    value = metric(*q)
    assert value.ndim == 0
    ref = calculate._metric_nets["lpips", "alex"](x, y)
    assert ref.item() == pytest.approx(0.10353731364011765, rel=1e-6)
    assert value.item() == pytest.approx(ref.item(), rel=1e-2)  # quantised to 8 bits
    assert abs(metric(q[0], q[0]).item()) <= 1e-9  # one batched pass: rounding only
    assert metric(q[1], q[0]).item() == pytest.approx(value.item(), rel=1e-6)


# ------------------------------------------------------------------ download-free LPIPS checks
def _untrained(monkeypatch, net):
    """lpips(net) on untrained torchvision features and seeded lin layers: no file is read."""
    from neosr.metrics import lpips as lp  # noqa: PLC0415

    build, _, ends, chns = lp.NETS[net]
    monkeypatch.setitem(lp.NETS, net, (build, None, ends, chns))
    g = torch.Generator().manual_seed(1)
    sd = {
        f"lin{k}.model.1.weight": torch.rand(1, c, 1, 1, generator=g)
        for k, c in enumerate(chns)
    }
    monkeypatch.setattr(
        torch.hub, "load_state_dict_from_url", lambda *_a, **_k: dict(sd)
    )
    torch.manual_seed(0)
    return lp.lpips(net).eval(), sd, ends


def _reference(model, sd, ends, x, y):
    """The v0.1 algorithm spelled out per image (two passes, taps after each ReLU end - 1)."""
    shift = torch.tensor([-0.030, -0.088, -0.188]).view(1, 3, 1, 1)
    scale = torch.tensor([0.458, 0.448, 0.450]).view(1, 3, 1, 1)
    layers = [m for s in model.slices for m in s]

    def taps(img):
        h, out = (img * 2 - 1 - shift) / scale, []
        for i, layer in enumerate(layers, 1):
            h = layer(h)
            if i in ends:
                out.append(h.clone())
        return out

    total = 0
    for k, (a, b) in enumerate(zip(taps(x), taps(y), strict=True)):
        na = a / (a.pow(2).sum(1, keepdim=True).sqrt() + 1e-10)
        nb = b / (b.pow(2).sum(1, keepdim=True).sqrt() + 1e-10)
        w = sd[f"lin{k}.model.1.weight"]
        total = total + ((na - nb) ** 2 * w).sum(1, keepdim=True).mean([2, 3])
    return total.squeeze()


@pytest.mark.parametrize("net", ["alex", "vgg"])
@pytest.mark.parametrize("shape", [(1, 3, 67, 53), (2, 3, 48, 40)])
def test_lpips_matches_the_spelled_out_algorithm(monkeypatch, net, shape):
    model, sd, ends = _untrained(monkeypatch, net)
    g = torch.Generator().manual_seed(2)
    x, y = torch.rand(shape, generator=g), torch.rand(shape, generator=g)
    with torch.inference_mode():
        torch.testing.assert_close(
            model(x, y), _reference(model, sd, ends, x, y), rtol=1e-4, atol=1e-6
        )


def test_lpips_grayscale_is_scored_as_rgb(monkeypatch):
    model, *_ = _untrained(monkeypatch, "alex")
    monkeypatch.setitem(calculate._metric_builders, "lpips", lambda _net: model)
    gray = quantize(torch.rand(1, 1, 40, 36)), quantize(torch.rand(1, 1, 40, 36))
    rgb = [t.expand(-1, 3, -1, -1) for t in gray]
    metric = build_metric({"type": "calculate_lpips"}, CPU)
    assert metric(*gray).item() == pytest.approx(metric(*rgb).item(), rel=1e-6)
    assert ("lpips", "alex") in calculate._metric_nets
