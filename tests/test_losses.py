import pytest
import torch

from neosr import losses as losses_pkg
from neosr.config import ConfigError
from neosr.losses import build_loss
from neosr.losses.base import Context

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
    if name == "kl_loss" and name not in losses_pkg.LOSSES:  # slated for deletion (G5.M6)
        pytest.skip("kl_loss not in the loss table")
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


@pytest.mark.parametrize(
    "name", ["L1Loss", "MSELoss", "HuberLoss", "chc_loss", "mssim_loss", "ncc_loss", "kl_loss"]
)
def test_compute_applies_weight(name):
    """forward is the unweighted kernel; compute(ctx) weights it once (and runs fp32 kernels in fp32)."""
    loss_fn = build(name)
    loss_fn.loss_weight = 0.5
    pred, target = make_pair("random")
    ref = loss_fn(pred, target) * 0.5
    sr = pred.clone().requires_grad_()
    value = loss_fn.compute(Context(sr=sr, gt=target, lq=target, it=0))
    torch.testing.assert_close(value, ref)
    value.backward()
    assert torch.isfinite(sr.grad).all()
    assert loss_fn.fp32 is (name == "mssim_loss")  # only mssim had custom_fwd(cast_inputs=float32)


@pytest.mark.parametrize("target", ["gt", "lq_up"])
@pytest.mark.parametrize("name", ["consistency_loss", "msswd_loss", "ldl_loss", "ff_loss"])
def test_compute_weights_kernel(name, target):
    """compute(ctx) at weight 0.5 == 0.5 * forward(sr, ref) at weight 1, in value and sr.grad.

    A kernel that still weights itself fails; msswd is reseeded per call; the fp32 kernels (ff,
    consistency since stage 2 L-02) are also checked under bf16 autocast against the fp32 kernel on
    the upcast input.
    """
    loss_fn = build(name)
    loss_fn.target = target
    assert loss_fn.fp32 is (name in {"ff_loss", "consistency_loss"})
    pred, gt = make_pair("random")
    lq = torch.rand(2, 3, 32, 32, device=DEV, generator=torch.Generator(DEV).manual_seed(1))
    ref_in = gt if target == "gt" else Context(sr=pred, gt=gt, lq=lq, it=0).lq_up
    for amp in (False, True) if loss_fn.fp32 else (False,):
        base = pred.bfloat16() if amp else pred
        x = base.detach().float().requires_grad_()
        loss_fn.loss_weight = 1.0
        torch.manual_seed(0)
        ref = loss_fn(x, ref_in) * 0.5
        ref.backward()
        sr = base.clone().requires_grad_()
        loss_fn.loss_weight = 0.5
        torch.manual_seed(0)
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=amp):
            value = loss_fn.compute(Context(sr=sr, gt=gt, lq=lq, it=0))
        value.backward()
        assert value.dtype == torch.float32
        torch.testing.assert_close(value, ref, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(sr.grad, x.grad.to(sr.dtype), rtol=1e-6, atol=1e-6)


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


PRE_B1 = "c48252d"  # last revision whose kernels still applied loss_weight themselves


def pre_b1_loss(name: str, file: str | None = None, **kwargs) -> torch.nn.Module:
    """The kernel as it was before B1, built from that revision's source (unregistered)."""
    import subprocess  # ruff: ignore[import-outside-top-level, suspicious-subprocess-import]
    import types  # ruff: ignore[import-outside-top-level]
    from pathlib import Path  # ruff: ignore[import-outside-top-level]

    root, path = Path(__file__).parents[1], f"neosr/losses/{file or name}.py"
    try:
        src = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            ["git", "show", f"{PRE_B1}:{path}"],  # ruff: ignore[start-process-with-partial-path]
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"pre-B1 source of {path} not available")
    mod = types.ModuleType(f"pre_b1_{file or name}")
    mod.__file__ = str(root / path)  # DISTS finds its weights next to the module
    src = src.replace("@LOSS_REGISTRY.register()\n", "").replace("from neosr.utils.registry import LOSS_REGISTRY\n", "")
    exec(compile(src, path, "exec"), mod.__dict__)  # ruff: ignore[exec-builtin]
    if "chc_loss" in mod.__dict__ and (file or name) != "basic_loss":  # VGG's criterion: pin it to pre-B1 too
        mod.chc_loss = type(pre_b1_loss("chc_loss", "basic_loss"))
    return getattr(mod, name)(**kwargs)


B1_W3 = {"fdl_loss": {"model": "vgg"}, "vgg_perceptual_loss": {}, "dists_loss": {}}


@pytest.mark.parametrize("amp", [False, True, "bf16-sr"])
@pytest.mark.parametrize("name", B1_W3)
def test_b1_matches_pre_b1(name, amp):
    """B1 is mechanical: compute(ctx) at weight w equals the old old(x, y) at loss_weight=w, values and grads.

    "bf16-sr" feeds a bf16 generator output (gt fp32) under autocast: custom_fwd's input cast must survive.
    """
    if amp and DEV != "cuda":
        pytest.skip("custom_fwd only acted under CUDA autocast")
    w, kw = 0.37, B1_W3[name]
    old = pre_b1_loss(name, loss_weight=w, **kw).to(DEV)
    new = build_loss({"type": name, **kw}).to(DEV)
    state = old.state_dict()
    new.load_state_dict({k: v for k, v in state.items() if not k.startswith("rand_")})
    for k, v in state.items():  # FDL projections are non-persistent now: copy them in
        if k.startswith("rand_"):
            getattr(new, k).copy_(v)
    new.loss_weight = w
    pred, target = make_pair("random")
    if amp == "bf16-sr":
        pred = pred.bfloat16()
    x_old, x_new = pred.clone().requires_grad_(), pred.clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(amp)):
        ref = old(x_old, target)
        value = new.compute(Context(sr=x_new, gt=target, lq=target, it=0))
    ref.backward()
    value.backward()
    assert value.dtype == ref.dtype == torch.float32
    assert x_new.grad.dtype == pred.dtype
    assert x_new.grad.abs().sum() > 0
    torch.testing.assert_close(value, ref, rtol=1e-6, atol=1e-6)
    # a bf16 grad is the fp32 one rounded: cuDNN backward is not bitwise deterministic, so old-vs-old
    # already differs by 1 bf16 ulp there (measured); compare at the bf16 default tolerance
    tol = {} if amp == "bf16-sr" else {"rtol": 1e-6, "atol": 1e-6}
    torch.testing.assert_close(x_new.grad, x_old.grad, **tol)


# id -> (pre-B1 file, build kwargs); old-vs-new is the B1 acceptance (D31) for the W1 kernels
B1_W1 = {
    "L1Loss": ("basic_loss", {}),
    "MSELoss": ("basic_loss", {}),
    "HuberLoss": ("basic_loss", {}),
    "HuberLoss-delta": ("basic_loss", {"delta": 0.1}),
    "chc_loss": ("basic_loss", {}),
    "chc_loss-l1": ("basic_loss", {"criterion": "l1", "loss_lambda": 0.02}),
    "mssim_loss": ("ssim_loss", {}),
    "ncc_loss": ("ncc_loss", {}),
    "kl_loss": ("kl_loss", {}),
}


@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("case", B1_W1)
def test_b1_w1_matches_pre_b1(case, amp):
    """compute(ctx) at weight w == old(x, y) at loss_weight=w, values and sr.grad, fp32 and CUDA bf16."""
    if amp and DEV != "cuda":
        pytest.skip("custom_fwd only acted under CUDA autocast")
    name, (file, kw), w = case.split("-")[0], B1_W1[case], 0.37
    old = pre_b1_loss(name, file, loss_weight=w, **kw).to(DEV)
    new = build_loss({"type": name, **kw}).to(DEV)
    if name == "mssim_loss":  # B1 mechanics only: the separable window has its own test (test_s2_losses)
        from tests.test_s2_losses import Gaussian2D  # ruff: ignore[import-outside-top-level]

        new.gaussian_filter = Gaussian2D().to(DEV)
    else:
        new.load_state_dict(old.state_dict())
    new.loss_weight = w
    pred, target = make_pair("random")
    x_old, x_new = pred.clone().requires_grad_(), pred.clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        ref = old(x_old, target)
        value = new.compute(Context(sr=x_new, gt=target, lq=target, it=0))
    ref.backward()
    value.backward()
    assert value.dtype == ref.dtype == torch.float32
    torch.testing.assert_close(value, ref, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(x_new.grad, x_old.grad, rtol=1e-6, atol=1e-6)


def test_fdl_projections_are_seeded():
    """D30: rand_* do not depend on the global RNG, are not checkpointed and follow .to()."""
    torch.manual_seed(1)
    a = build_loss({"type": "fdl_loss", "model": "vgg"})
    torch.manual_seed(2)
    b = build_loss({"type": "fdl_loss", "model": "vgg"}).to(DEV)
    c = build_loss({"type": "fdl_loss", "model": "vgg", "seed": 1})
    rand = [k for k, _ in a.named_buffers() if k.startswith("rand_")]
    assert len(rand) == len(a.model.chns)
    assert all(torch.equal(getattr(a, k), getattr(b, k).cpu()) for k in rand)
    assert not any(torch.equal(getattr(a, k), getattr(c, k)) for k in rand)
    assert all(getattr(b, k).device.type == DEV for k in rand)
    assert not any(k.startswith("rand_") for k in a.state_dict())
    with torch.device(DEV):  # a device context must not move the draw off the CPU generator
        d = build_loss({"type": "fdl_loss", "model": "vgg"})
    assert all(torch.equal(getattr(a, k), getattr(d, k).cpu()) for k in rand)
    norms = a.rand_0.flatten(1).norm(dim=1)
    torch.testing.assert_close(norms, torch.ones_like(norms))


