import pytest
import torch

from neosr.config import ConfigError

from neosr.losses import build_loss
from neosr.utils.registry import LOSS_REGISTRY

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SHAPE = (2, 3, 64, 64)

# name -> (build kwargs, max loss on identical inputs or None if not meaningful)
LOSSES = {
    "L1Loss": ({}, 1e-3),
    "MSELoss": ({}, 1e-3),
    "HuberLoss": ({}, 1e-3),
    "chc_loss": ({}, 5e-3),  # floored at clip_min (~0.0039) by design
    "mssim_loss": ({}, 1e-3),
    "ncc_loss": ({}, 1e-3),
    "kl_loss": ({}, 1e-3),
    "consistency_loss": ({}, 1e-3),
    "msswd_loss": ({}, 1e-3),
    "ldl_loss": ({}, 1e-3),
    "ff_loss": ({}, 1e-3),
    "vgg_perceptual_loss": ({}, 1e-3),
    "dists_loss": ({}, 1e-3),
    "fdl_loss": ({"model": "vgg"}, 1e-3),
}
# "offset": nonzero residual with zero variance (constant pred - target)
CASES = ("random", "identical", "constant", "offset", "anticorrelated", "channels_last")


def make_pair(case: str, dev: str = DEV) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=dev).manual_seed(0)
    pred = torch.rand(SHAPE, device=dev, generator=g)
    target = torch.rand(SHAPE, device=dev, generator=g)
    if case == "identical":
        target = pred.clone()
    elif case == "constant":
        pred, target = torch.full(SHAPE, 0.5, device=dev), torch.full(SHAPE, 0.5, device=dev)
    elif case == "offset":
        pred = pred * 0.8
        target = pred + 0.1
    elif case == "anticorrelated":
        yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
        board = ((yy + xx) % 2).float().to(dev).expand(SHAPE).contiguous()
        pred, target = board, 1.0 - board
    elif case == "channels_last":
        pred = pred.contiguous(memory_format=torch.channels_last)
    return pred, target


def build(name: str, dev: str = DEV) -> torch.nn.Module:
    if name == "kl_loss" and name not in LOSS_REGISTRY:  # slated for deletion (G5.M6)
        pytest.skip("kl_loss not registered")
    return build_loss({"type": name, **LOSSES[name][0]}).to(dev)


@pytest.fixture(scope="module")
def losses():
    cache: dict[str, torch.nn.Module] = {}

    def get(name: str) -> torch.nn.Module:
        if name not in cache:
            cache[name] = build(name)
        return cache[name]

    return get


def check_forward_backward(loss_fn, pred, target, max_identical=None):
    pred = pred.requires_grad_()
    loss = loss_fn(pred, target)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    if max_identical is not None:
        assert loss.abs().item() <= max_identical
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("name", LOSSES)
def test_loss_forward_backward(losses, name, case):
    pred, target = make_pair(case)
    max_identical = LOSSES[name][1] if case == "identical" else None
    check_forward_backward(losses(name), pred, target, max_identical)


@pytest.mark.parametrize("name", LOSSES)
def test_loss_on_cpu(name):
    """No hidden dependency on CUDA or on torch.set_default_device."""
    check_forward_backward(build(name, "cpu"), *make_pair("random", "cpu"))


def test_dists_metric_path():
    metric = build_loss({"type": "dists_loss", "as_loss": False}).to("cpu")
    pred, target = make_pair("random", "cpu")
    with torch.no_grad():
        assert torch.isfinite(metric(pred[:1], target[:1]))
        assert metric(pred[:1], pred[:1]).abs() < 1e-3
    assert not any(p.requires_grad for p in metric.parameters())


def test_chc_reduction_and_fp16():
    pred, target = make_pair("random")
    kw = {"type": "chc_loss", "clip_min": 0, "clip_max": float("inf")}
    elem = build_loss({**kw, "reduction": "none"})(pred, target)
    assert elem.shape == SHAPE
    assert build_loss({**kw, "reduction": "sum"})(pred, target).item() == pytest.approx(
        elem.sum().item(), rel=1e-5
    )
    # charbonnier eps underflows in fp16 unless computed in fp32
    p = pred.half().requires_grad_()
    build_loss(kw)(p, p.detach()).backward()
    assert torch.isfinite(p.grad).all()
    # the cosine term must survive zero vectors in fp16 too
    z = torch.zeros(SHAPE, device=DEV, dtype=torch.half, requires_grad=True)
    loss = build_loss({"type": "chc_loss", "loss_lambda": 0.02})(z, z.detach())
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(z.grad).all()
    # fp64 keeps its precision
    d = torch.full(SHAPE, 0.5, device=DEV, dtype=torch.float64, requires_grad=True)
    loss = build_loss({**kw, "criterion": "l1"})(d, d.detach() + 1e-8)
    loss.backward()
    assert loss.item() == pytest.approx(1e-8, rel=1e-3)
    assert d.grad.abs().sum() > 0


def test_ncc_per_sample_channel():
    pred, _ = make_pair("random")
    # perfectly correlated per (N, C), but each slice has its own offset
    target = pred * 2 + torch.arange(6, device=DEV, dtype=pred.dtype).view(2, 3, 1, 1)
    loss_fn = build("ncc_loss")
    assert loss_fn(pred, target).item() == pytest.approx(0, abs=1e-5)
    target = torch.rand_like(pred)
    singles = [
        loss_fn(pred[n : n + 1, c : c + 1], target[n : n + 1, c : c + 1])
        for n in range(2)
        for c in range(3)
    ]
    assert loss_fn(pred, target).item() == pytest.approx(
        torch.stack(singles).mean().item(), rel=1e-5
    )


def test_msswd_averages_all_scales(monkeypatch):
    loss_fn = build("msswd_loss")
    # a per-scale value equal to the scale's width: 64, 32, 16
    monkeypatch.setattr(
        loss_fn, "forward_once", lambda x, _y: x.new_full((x.shape[0],), x.shape[-1])
    )
    pred, target = make_pair("random")
    assert loss_fn(pred, target).item() == pytest.approx(0.1 * (64 + 32 + 16) / 3)


def test_consistency_l_star_low_branch():
    loss_fn = build("consistency_loss")
    # sRGB 0.02 -> linear Y 0.02/12.92, below the CIE knee: L* = Y * 24389/27
    img = torch.full((1, 3, 4, 4), 0.02, device=DEV)
    expected = 0.02 / 12.92 * 24389 / 27 / 100
    assert loss_fn.rgb_to_l_star(img).mean().item() == pytest.approx(expected, rel=1e-3)


def test_frozen_weights_and_options():
    assert not any(p.requires_grad for p in build("dists_loss", "cpu").parameters())
    assert build_loss({"type": "fdl_loss", "model": "VGG"}).model_name == "vgg"
    with pytest.raises(ConfigError):  # PatchLoss/IPK were removed (G5.4)
        build_loss({"type": "vgg_perceptual_loss", "patchloss": True})
    with pytest.raises(ValueError, match="vgg19"):
        build_loss({"type": "vgg_perceptual_loss", "vgg_type": "vgg16"})


@pytest.mark.parametrize("target_is_real", [True, False])
@pytest.mark.parametrize("gan_type", ["bce", "mse", "huber"])
def test_gan_loss(gan_type, target_is_real):
    weight = 0.25
    loss_fn = build_loss({"type": "gan_loss", "gan_type": gan_type, "loss_weight": weight})
    logits = torch.randn(4, 1, 8, 8, device=DEV, requires_grad=True)
    d_loss = loss_fn(logits, target_is_real, is_disc=True)
    g_loss = loss_fn(logits, target_is_real, is_disc=False)
    assert torch.isfinite(d_loss)
    assert torch.isfinite(g_loss)
    assert g_loss.item() == pytest.approx(d_loss.item() * weight, rel=1e-5)
    g_loss.backward()
    assert torch.isfinite(logits.grad).all()
