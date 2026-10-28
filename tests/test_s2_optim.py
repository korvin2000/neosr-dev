"""Stage 2, optimizer lane: sound schedule-free Adan (O-01, O-03, O-04, O-05, O-06, O-07, O-08)."""

import io
import math

import pytest
import torch
from torch.testing import assert_close
from torch.utils._python_dispatch import TorchDispatchMode

from neosr.config import ConfigError
from neosr.optimizers import adamw_sf, adan, adan_sf

SHAPES = ((4, 3), (7,), (2, 3, 2))
SF_STATE = {"z", "exp_avg_sq", "exp_avg_diff", "neg_pre_grad"}
ADAN_STATE = {"exp_avg", "exp_avg_sq", "exp_avg_diff", "neg_pre_grad"}
# group keys of a schedule_free=False adan_sf state, as stage 1 wrote them
OLD_NOSF_GROUP = {
    "lr",
    "betas",
    "eps",
    "r",
    "weight_decay",
    "max_grad_norm",
    "warmup_steps",
    "train_mode",
    "weight_sum",
    "lr_max",
    "weight_lr_power",
    "schedule_free",
    "step",
    "params",
}


def _params(dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [
        torch.nn.Parameter(torch.randn(s, generator=g, dtype=dtype)) for s in SHAPES
    ]


def _grads(t, ys):
    """Noisy ill-conditioned quadratic: g = h * y + noise_t, a function of (t, y) only."""
    gen = torch.Generator().manual_seed(1000 + t)
    out = []
    for y in ys:
        h = torch.linspace(0.1, 10.0, y.numel(), dtype=y.dtype).view(y.shape)
        noise = torch.randn(y.shape, generator=gen, dtype=y.dtype)
        out.append(h.to(y.device) * y + 0.3 * noise.to(y.device))
    return out


def _set_grads(ps, t):
    for p, g in zip(ps, _grads(t, [p.detach() for p in ps]), strict=True):
        p.grad = g


def _run(opt, ps, steps, start=1):
    for t in range(start, start + steps):
        _set_grads(ps, t)
        opt.step()


def _clone(ps):
    return [torch.nn.Parameter(p.detach().clone()) for p in ps]


def _roundtrip(sd):
    """Through bytes, as a checkpoint: an in-memory state_dict shares its tensors."""
    buf = io.BytesIO()
    torch.save(sd, buf)
    buf.seek(0)
    return torch.load(buf, weights_only=True)


def _reference(p0s, steps, *, lr, betas, eps, wd, warmup, r=0.0, wlp=2.0):
    """The recurrence in its explicit x/y/z form, per tensor, nothing in place."""
    b1, b2, b3 = betas
    x = [p.detach().clone() for p in p0s]
    z = [p.detach().clone() for p in p0s]
    v = [torch.zeros_like(p) for p in x]
    n = [torch.zeros_like(p) for p in x]
    g_prev, lr_max, w_sum = None, -1.0, 0.0
    for t in range(1, steps + 1):
        y = [(1 - b1) * zi + b1 * xi for zi, xi in zip(z, x, strict=True)]
        g = _grads(t, y)
        g_prev = g if g_prev is None else g_prev  # d_1 = 0
        gamma = lr * (min(t / warmup, 1.0) if warmup else 1.0) * math.sqrt(1 - b3**t)
        lr_max = max(lr_max, gamma)
        w = t**r * lr_max**wlp
        w_sum += w
        c = w / w_sum
        for i in range(len(x)):
            d = g[i] - g_prev[i]
            v[i] = b2 * v[i] + (1 - b2) * d
            u = g[i] + b2 * d
            n[i] = b3 * n[i] + (1 - b3) * u**2
            U = (g[i] + b2 / (1 - b2**t) * v[i]) / (n[i].sqrt() + eps) + wd * y[i]
            z[i] = z[i] - gamma * U
            x[i] = (1 - c) * x[i] + c * z[i]
        g_prev = g
    y = [(1 - b1) * zi + b1 * xi for zi, xi in zip(z, x, strict=True)]
    return x, y, z


@pytest.mark.parametrize("b2", [0.92, 0.0])
def test_sf_kernel_matches_explicit_reference(b2):
    kw = {"lr": 1e-2, "betas": (0.98, b2, 0.99), "eps": 1e-8}
    ps = _params(torch.float64)
    x_ref, y_ref, z_ref = _reference(ps, 300, wd=0.01, warmup=20, **kw)
    opt = adan_sf(ps, weight_decay=0.01, warmup_steps=20, **kw)
    _run(opt, ps, 300)
    for p, y, z in zip(ps, y_ref, z_ref, strict=True):
        assert_close(p.detach(), y, rtol=1e-9, atol=1e-11)
        assert_close(opt.state[p]["z"], z, rtol=1e-9, atol=1e-11)
    opt.eval()
    for p, x in zip(ps, x_ref, strict=True):
        assert_close(p.detach(), x, rtol=1e-9, atol=1e-11)


def test_b2_zero_is_adamw_sf_bitwise():
    a, b = _params(), _params()
    kw = {"lr": 1e-2, "eps": 1e-8, "weight_decay": 0.01, "warmup_steps": 10}
    oa = adan_sf(a, betas=(0.98, 0.0, 0.99), **kw)
    ob = adamw_sf(b, betas=(0.98, 0.99), **kw)
    for t in range(1, 61):
        for opt, ps in ((oa, a), (ob, b)):
            _set_grads(ps, t)
            opt.step()
    for p, q in zip(a, b, strict=True):
        assert torch.equal(p, q)
        assert set(oa.state[p]) == {"z", "exp_avg_sq"}
        for k, buf in oa.state[p].items():
            assert torch.equal(buf, ob.state[q][k])
    oa.eval()
    ob.eval()
    assert all(torch.equal(p, q) for p, q in zip(a, b, strict=True))


@pytest.mark.parametrize("b2", [0.92, 0.0])
def test_first_step_x_y_z_coincide(b2):
    ps = _params()
    opt = adan_sf(
        ps, lr=1e-2, betas=(0.98, b2, 0.99), weight_decay=0.01, warmup_steps=5
    )
    _run(opt, ps, 1)
    y = [p.detach().clone() for p in ps]
    opt.eval()
    for p, yi in zip(ps, y, strict=True):
        assert torch.equal(opt.state[p]["z"], yi)
        assert torch.equal(p.detach(), yi)


@pytest.mark.parametrize("warmup", [0, 10])
@pytest.mark.parametrize("b2", [0.92, 0.0])
def test_scalar_first_step(b2, warmup):
    lr, b3, eps = 1e-2, 0.99, 1e-8
    p = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
    opt = adan_sf(
        [p], lr=lr, betas=(0.98, b2, b3), eps=eps, weight_decay=0.0, warmup_steps=warmup
    )
    p.grad = torch.ones(1, dtype=torch.float64)
    opt.step()
    eta, q = lr * (1 / warmup if warmup else 1.0), math.sqrt(1 - b3)
    expected = torch.tensor([1 - eta * q / (q + eps)], dtype=torch.float64)
    assert_close(p.detach(), expected)
    assert_close(opt.state[p]["z"], expected)
    opt.eval()
    assert_close(p.detach(), expected)


def test_zero_gradient_weight_decay_first_step():
    lr, wd, b3 = 0.1, 0.1, 0.99
    ps = _params(torch.float64)
    p0 = [p.detach().clone() for p in ps]
    opt = adan_sf(ps, lr=lr, betas=(0.98, 0.92, b3), weight_decay=wd)
    for p in ps:
        p.grad = torch.zeros_like(p)
    opt.step()
    k = (
        1 - lr * math.sqrt(1 - b3) * wd
    )  # 1 - gamma_1 * wd = 0.999: decay survives step 1
    for p, t0 in zip(ps, p0, strict=True):
        assert_close(p.detach(), k * t0)
        assert_close(opt.state[p]["z"], k * t0)
    opt.eval()
    for p, t0 in zip(ps, p0, strict=True):
        assert_close(p.detach(), k * t0)


def test_eval_moves_downhill_on_quadratic():
    gen = torch.Generator().manual_seed(0)
    a = torch.linspace(0.5, 5.0, 12).view(4, 3)
    target = torch.randn(4, 3, generator=gen)
    p = torch.nn.Parameter(torch.zeros(4, 3))

    def f(w):
        return float(0.5 * (a * (w - target) ** 2).sum())

    f0, losses = f(p.detach()), []
    opt = adan_sf([p], lr=5e-2, weight_decay=0.0, warmup_steps=10)
    for _ in range(300):
        p.grad = a * (p.detach() - target)
        opt.step()
        opt.eval()
        losses.append(f(p.detach()))
        opt.train()
    assert losses[0] < f0  # the stage-1 recurrence moved the eval weights uphill here
    assert losses[-1] < 1e-2 * f0
    assert losses[-1] < losses[len(losses) // 2] < losses[0]


@pytest.mark.parametrize("make", [adan_sf, adamw_sf])
def test_eval_train_is_one_lerp_of_the_old_formula(make):
    ps = _params()
    frozen = ps[-1]  # never gets a gradient: no z, must not move
    opt = make(ps, lr=1e-2)
    for t in range(1, 4):
        _set_grads(ps[:-1], t)
        opt.step()
    beta1 = opt.param_groups[0]["betas"][0]
    y = [p.detach().clone() for p in ps]
    opt.eval()
    opt.eval()  # idempotent
    for p, yi in zip(ps[:-1], y[:-1], strict=True):
        assert torch.equal(p.detach(), yi.lerp(opt.state[p]["z"], 1 - 1 / beta1))
    assert torch.equal(frozen.detach(), y[-1])
    assert "z" not in opt.state[frozen]
    x = [p.detach().clone() for p in ps]
    opt.train()
    for p, xi in zip(ps[:-1], x[:-1], strict=True):
        assert torch.equal(p.detach(), xi.lerp(opt.state[p]["z"], 1 - beta1))
    assert all(g["train_mode"] for g in opt.param_groups)


@pytest.mark.parametrize("b2", [0.92, 0.0])
def test_state_dict_resume_is_exact(b2):
    kw = {"lr": 1e-2, "betas": (0.98, b2, 0.99), "weight_decay": 0.01}
    a = _params()
    oa = adan_sf(a, warmup_steps=5, **kw)
    _run(oa, a, 7)
    sd = _roundtrip(oa.state_dict())
    assert [g["sf_version"] for g in sd["param_groups"]] == [adan_sf.STATE_VERSION]
    b = _clone(a)
    ob = adan_sf(b, warmup_steps=5, **kw)
    ob.load_state_dict(sd)
    _run(oa, a, 5, start=8)
    _run(ob, b, 5, start=8)
    for p, q in zip(a, b, strict=True):
        assert torch.equal(p, q)
        assert oa.state[p].keys() == ob.state[q].keys()
        for k, buf in oa.state[p].items():
            assert torch.equal(buf, ob.state[q][k])


def _sf_state_dict():
    ps = _params()
    opt = adan_sf(ps, lr=1e-2)
    _run(opt, ps, 2)
    return opt.state_dict()


def _refused(sd, *parts):
    opt = adan_sf(_params(seed=1), lr=1e-2)
    with pytest.raises(ConfigError) as e:
        opt.load_state_dict(sd)
    msg = str(e.value)
    assert all(s in msg for s in ("adan_sf", "pretrain_network_", *parts)), msg
    # refused before anything was replaced
    assert not opt.state
    assert opt.param_groups[0]["sf_version"] == adan_sf.STATE_VERSION
    assert "step" not in opt.param_groups[0]


@pytest.mark.parametrize("version", [None, 1, 3])
def test_old_sf_state_refused(version):
    sd = _sf_state_dict()
    for g in sd["param_groups"]:
        if version is None:  # stage-1 states carry no version
            del g["sf_version"]
        else:
            g["sf_version"] = version
    _refused(sd, "version")


def test_stage1_sf_layout_refused():
    sd = _sf_state_dict()
    del sd["param_groups"][0]["sf_version"]
    del sd["param_groups"][0]["schedule_free"]  # the oldest states: default True
    for s in sd["state"].values():
        s["exp_avg"] = torch.zeros_like(s["z"])
    _refused(sd, "version")


@pytest.mark.parametrize("key", sorted(SF_STATE))
def test_sf_state_missing_buffer_refused(key):
    sd = _sf_state_dict()
    del sd["state"][1][key]
    _refused(sd, key, "#1")


def test_old_non_sf_state_accepted():
    kw = {"lr": 1e-2, "warmup_steps": 4, "schedule_free": False}
    a = _params()
    oa = adan_sf(a, **kw)
    _run(oa, a, 6)
    sd = _roundtrip(oa.state_dict())
    # same layout as stage 1 wrote: no sf_version, the four Adan buffers
    assert set(sd["param_groups"][0]) == OLD_NOSF_GROUP
    assert all(set(s) == ADAN_STATE for s in sd["state"].values())
    b = _clone(a)
    ob = adan_sf(b, **kw)
    ob.load_state_dict(sd)
    _run(oa, a, 4, start=7)
    _run(ob, b, 4, start=7)
    assert all(torch.equal(p, q) for p, q in zip(a, b, strict=True))


def _stage1_adan_step(ps, state, step, *, lr, betas, eps, wd):
    """The stage-1 adan kernel (no_prox), per tensor, ops in their original order."""
    b1, b2, b3 = betas
    bc1, bc2, bc3_sqrt = 1.0 - b1**step, 1.0 - b2**step, math.sqrt(1.0 - b3**step)
    for p in ps:
        s = state.setdefault(p, {})
        if not s:
            s.update({k: torch.zeros_like(p) for k in ("m", "n", "v")})
        g = p.grad
        if "neg" not in s or step == 1:
            s["neg"] = g.clone().mul_(-1.0)
        neg = s["neg"]
        g.mul_(1.0)
        neg.add_(g)
        s["m"].mul_(b1).add_(g, alpha=1 - b1)
        s["v"].mul_(b2).add_(neg, alpha=1 - b2)
        neg.mul_(b2).add_(g)
        s["n"].mul_(b3).addcmul_(neg, neg, value=1 - b3)
        denom = (s["n"].sqrt() / bc3_sqrt).add_(eps)
        with torch.no_grad():
            p.mul_(1 - lr * wd)
            p.addcdiv_(s["m"], denom, value=-(lr / bc1))
            p.addcdiv_(s["v"], denom, value=-(lr * b2 / bc2))
        neg.zero_().add_(g, alpha=-1.0)


@pytest.mark.parametrize(
    ("make", "warmup"),
    [
        (lambda ps, **kw: adan(ps, no_prox=True, **kw), 0),
        (lambda ps, **kw: adan(ps, no_prox=True, foreach=False, **kw), 0),
        (lambda ps, **kw: adan_sf(ps, schedule_free=False, **kw), 0),
        (lambda ps, **kw: adan_sf(ps, schedule_free=False, warmup_steps=4, **kw), 4),
    ],
)
def test_plain_adan_matches_stage1_kernel(make, warmup):
    lr, betas, eps, wd = 1e-2, (0.98, 0.92, 0.99), 1e-8, 0.02
    ref, ref_state = _params(), {}
    ps = _params()
    opt = make(ps, lr=lr, betas=betas, eps=eps, weight_decay=wd)
    for t in range(1, 31):
        _set_grads(ref, t)
        sched = t / warmup if t < warmup else 1.0
        _stage1_adan_step(ref, ref_state, t, lr=lr * sched, betas=betas, eps=eps, wd=wd)
        _set_grads(ps, t)
        opt.step()
    assert all(torch.equal(p, q) for p, q in zip(ps, ref, strict=True))


@pytest.mark.parametrize(
    ("kw", "keys"),
    [
        ({}, SF_STATE),
        ({"betas": (0.98, 0.0, 0.99)}, {"z", "exp_avg_sq"}),
        ({"schedule_free": False}, ADAN_STATE),
    ],
)
def test_state_buffers(kw, keys):
    ps = _params()
    opt = adan_sf(ps, lr=1e-2, **kw)
    _run(opt, ps, 2)
    assert all(set(opt.state[p]) == keys for p in ps)


class _Ops(TorchDispatchMode):
    """Top-level aten ops (name, scalar arg); a _foreach op is one launch group on CUDA."""

    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        scalar = args[1] if len(args) > 1 and isinstance(args[1], float) else None
        self.ops.append((func.overloadpacket.__name__, scalar))
        return func(*args, **(kwargs or {}))


def _ops(make, n_tensors, call="step"):
    gen = torch.Generator().manual_seed(0)
    ps = [
        torch.nn.Parameter(torch.randn(3, 2, generator=gen)) for _ in range(n_tensors)
    ]
    opt = make(ps)
    for _ in range(2):  # the first step initialises the state
        for p in ps:
            p.grad = torch.randn(3, 2, generator=gen)
        with _Ops() as rec:
            getattr(opt, call)()
    return rec.ops


SF_B2 = {"lr": 1e-2, "weight_decay": 0.01}
SF_B0 = {**SF_B2, "betas": (0.98, 0.0, 0.99)}


@pytest.mark.parametrize(
    ("make", "count"),
    [
        (lambda ps: adan_sf(ps, **SF_B2), 16),  # stage 1: 20
        (lambda ps: adan_sf(ps, **SF_B0), 9),
        (lambda ps: adan_sf(ps, schedule_free=False), 17),  # stage 1: 18
        (lambda ps: adan_sf(ps, max_grad_norm=1.0, **SF_B2), None),
        (lambda ps: adan(ps, max_grad_norm=1.0), None),
    ],
)
def test_step_launches_do_not_scale_with_tensors(make, count):
    ops = _ops(make, 10)
    assert ops == _ops(make, 40)  # no per-tensor launch (O-07: one _foreach_norm)
    assert ("_foreach_mul_", 1.0) not in ops  # O-05: no grads *= 1.0 pass
    if count is not None:
        assert len(_foreach(ops)) == count, ops
        assert len(ops) == count + 2  # + the optimizer's record_function enter/exit


def _foreach(ops):
    return [op for op in ops if op[0].startswith("_foreach")]


def test_b2_zero_launches_equal_adamw_sf():
    adamw = _ops(lambda ps: adamw_sf(ps, betas=(0.98, 0.99), **SF_B2), 10)
    assert _foreach(_ops(lambda ps: adan_sf(ps, **SF_B0), 10)) == _foreach(adamw)


@pytest.mark.parametrize("make", [adan_sf, adamw_sf])
@pytest.mark.parametrize("call", ["eval", "train"])
def test_swap_is_one_lerp_per_group(make, call):
    for n in (10, 40):
        gen = torch.Generator().manual_seed(0)
        ps = [torch.nn.Parameter(torch.randn(3, 2, generator=gen)) for _ in range(n)]
        opt = make(ps, lr=1e-2)
        for p in ps:
            p.grad = torch.ones_like(p)
        opt.step()
        if call == "train":
            opt.eval()
        with _Ops() as rec:
            getattr(opt, call)()
        assert [name for name, _ in rec.ops] == ["_foreach_lerp_"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_b2_zero_is_adamw_sf_and_swap_is_the_lerp():
    a = [torch.nn.Parameter(p.detach().cuda()) for p in _params()]
    b = _clone(a)
    kw = {"lr": 1e-2, "weight_decay": 0.01, "warmup_steps": 3}
    oa = adan_sf(a, betas=(0.98, 0.0, 0.99), **kw)
    ob = adamw_sf(b, betas=(0.98, 0.99), **kw)
    for t in range(1, 11):
        for opt, ps in ((oa, a), (ob, b)):
            _set_grads(ps, t)
            opt.step()
    assert all(torch.equal(p, q) for p, q in zip(a, b, strict=True))
    y = [p.detach().clone() for p in a]
    oa.eval()
    for p, yi in zip(a, y, strict=True):
        assert torch.equal(p.detach(), yi.lerp(oa.state[p]["z"], 1 - 1 / 0.98))
