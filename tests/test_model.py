"""Trainer integration (design T2 §9 C1 (i)-(v), T1 §6 B2 checks that need no train.py), on tiny CPU nets."""

from contextlib import contextmanager
from itertools import starmap
from pathlib import Path

import cv2
import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

from neosr.config import ConfigError
from neosr.models import build_model
from neosr.models.image import QAT_KEYS
from neosr.runtime import Runtime

CPU = Runtime(torch.device("cpu"), 0, 1, 0, False)
S, P, B = 2, 8, 2  # scale, lq patch, batch
STATE_KEYS = {
    "format",
    "update",
    "epoch",
    "nets",
    "ema",
    "phases",
    "rng",
    "best",
    "run",
}
G_KEYS = {"l_g_pix", "l_g_total", "grad_norm_g", "clipped_g", "skipped_g"}
D_KEYS = {
    "l_g_gan",
    "l_d_real",
    "l_d_fake",
    "out_d_real",
    "out_d_fake",
    "l_d_total",
    "grad_norm_d",
    "clipped_d",
    "skipped_d",
}
SF = {"type": "adan_sf", "lr": 1e-2, "schedule_free": True}


def make_opt(
    tmp,
    *,
    d=False,
    ema=0.0,
    optim=None,
    accumulate=1,
    is_train=True,
    model_type="image",
):
    opt = {
        "name": "tiny",
        "model_type": model_type,
        "scale": S,
        "is_train": is_train,
        "network_g": {"type": "compact", "num_feat": 8, "num_conv": 2},
        "datasets": {},
        "val": {
            "pbar": False,
            "metrics": {"psnr": {"type": "calculate_psnr", "crop_border": 0}},
        },
        "logger": {},
        "path": {
            "models": tmp / "models",
            "training_states": tmp / "training_states",
            "visualization": tmp / "vis",
        },
    }
    if not is_train:
        return opt
    opt["datasets"]["train"] = {
        "name": "t",
        "type": "paired",
        "patch_size": P,
        "batch_size": B,
        "accumulate": accumulate,
    }
    opt["train"] = {
        "optim_g": optim or {"type": "adamw", "lr": 1e-3},
        "pixel_opt": {"type": "L1Loss", "loss_weight": 1.0},
        "ema": ema,
    }
    if d:
        opt["network_d"] = {"type": "unet", "num_feat": 4}  # spectral norm: D buffers
        opt["train"] |= {
            "optim_d": {"type": "adamw", "lr": 1e-4},
            "gan_opt": {"type": "gan_loss", "loss_weight": 0.1},
        }
    return opt


def build(opt, rt=CPU, seed=0):
    torch.manual_seed(seed)
    return build_model(opt, rt)


def batches(n, b=B, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [
        {
            "lq": torch.rand(b, 3, P, P, generator=g),
            "gt": torch.rand(b, 3, P * S, P * S, generator=g),
        }
        for _ in range(n)
    ]


def run(model, data):
    """Feed each batch and step; returns optimize_parameters() per micro-batch."""
    done = []
    for x in data:
        model.feed_data(x)
        done.append(model.optimize_parameters())
    return done


class ValSet(Dataset):
    def __init__(self, kind="paired", n=2, h=12, w=10, name="valset") -> None:
        self.opt = {"name": name, "type": kind}
        g = torch.Generator().manual_seed(7)
        self.items = [
            (torch.rand(3, h, w, generator=g), torch.rand(3, h * S, w * S, generator=g))
            for _ in range(n)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        lq, gt = self.items[i]
        return {"lq": lq, "lq_path": f"/x/img{i}.png"} | (
            {} if self.opt["type"] == "single" else {"gt": gt}
        )


def loader(**kw):
    return DataLoader(ValSet(**kw), batch_size=1)


def clone(net):
    return [t.detach().clone() for t in (*net.parameters(), *net.buffers())]


def same(a, b):
    return len(a) == len(b) and all(starmap(torch.equal, zip(a, b, strict=True)))


def sd_equal(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


def opt_tensors(opt):
    return [
        v.clone()
        for s in opt.state_dict()["state"].values()
        for v in s.values()
        if torch.is_tensor(v)
    ]


# ---------------------------------------------------------------------------- (i) inference path
@pytest.mark.parametrize("model_type", ["image", "otf"])
def test_inference_path_writes_test_layout(tmp_path, model_type):
    train = build(make_opt(tmp_path / "src"))
    sd = train.export_state_dicts()["net_g"]
    torch.save({"params": sd}, tmp_path / "g.pth")
    opt = make_opt(tmp_path, is_train=False, model_type=model_type)
    opt["path"]["pretrain_network_g"] = str(tmp_path / "g.pth")
    m = build(opt, seed=1)
    assert m.net_d is None
    assert m.ema is None
    assert m.phases == {}
    assert m.net_g_fwd is None
    assert not hasattr(m, "degrader")
    assert not hasattr(m, "g")
    assert sd_equal(m.net_g.state_dict(), sd)  # the pretrained weights, not seed 1
    m.validation(loader(), current_iter="name", tb_logger=None, save_img=True)
    vis = tmp_path / "vis"
    assert sorted(p.name for p in (vis / "valset").iterdir()) == [
        "img0_tiny.png",
        "img1_tiny.png",
    ]
    assert (vis / "img0" / "img0_lq.png").is_file()
    assert set(m.best_metric_results["valset"]) == {"psnr"}
    m.opt["val"]["suffix"] = "sfx"  # single: no gt, no metrics
    m.validation(
        loader(kind="single", name="single"),
        current_iter="name",
        tb_logger=None,
        save_img=True,
    )
    assert sorted(p.name for p in (vis / "single").iterdir()) == [
        "img0_sfx.png",
        "img1_sfx.png",
    ]
    assert "single" not in m.best_metric_results
    assert m.net_g.training  # modes restored


def test_tiled_validation_matches_full(tmp_path):
    opt = make_opt(tmp_path, is_train=False)
    opt["val"] |= {"suffix": "full"}
    m = build(opt)
    m.validation(loader(h=20, w=24), "x", None)
    m.opt["val"]["suffix"], m.tile, m.tile_overlap = (
        "tiled",
        12,
        4,
    )  # halo >= RF of 4 3x3 convs: exact
    m.validation(loader(h=20, w=24), "x", None)
    for i in range(2):
        a, b = (
            cv2.imread(str(tmp_path / "vis" / "valset" / f"img{i}_{s}.png")).astype(int)
            for s in ("full", "tiled")
        )
        assert a.shape == (40, 48, 3)
        assert abs(a - b).max() <= 1


def test_unknown_model_type(tmp_path):
    opt = make_opt(tmp_path, is_train=False)
    opt["model_type"] = "imagee"
    with pytest.raises(ConfigError, match="did you mean 'image'"):
        build(opt)


# ---------------------------------------------------------------------------- training step (T1 B2)
@pytest.mark.parametrize("d", [False, True], ids=["G", "GD"])
@pytest.mark.parametrize("accumulate", [1, 2])
@pytest.mark.parametrize("ema", [0.0, 0.9])
def test_train_step(tmp_path, d, accumulate, ema):
    m = build(make_opt(tmp_path, d=d, ema=ema, accumulate=accumulate))
    assert (m.ema is not None) == (ema > 0)
    assert set(m.phases) == ({"net_g", "net_d"} if d else {"net_g"})
    if d:
        assert m.net_d.training
        assert m.net_d_adv is m.net_d_fwd is m.net_d
    its, term = [], m.g.terms[0]
    orig = term.compute
    term.compute = lambda ctx: its.append(ctx.it) or orig(ctx)
    g0 = clone(m.net_g)
    done = run(m, batches(2 * accumulate))
    assert done == [(i + 1) % accumulate == 0 for i in range(2 * accumulate)]
    assert (m.micro, m.update) == (2 * accumulate, 2)
    assert its == [u for u in (1, 2) for _ in range(accumulate)]  # it = update + 1
    log = m.get_current_log()
    assert set(log) == G_KEYS | (D_KEYS if d else set())
    assert log["skipped_g"] == 0
    assert all(v == v for v in log.values())
    assert not same(g0, clone(m.net_g))
    assert all(p.grad is None for p in m.net_g.parameters())
    if ema:  # first update copied, second lerped: EMA lags the live weights
        assert not same(clone(m.ema.module), clone(m.net_g))
    m.check_finite()
    assert m.get_current_log() == {"skipped_g": 0} | ({"skipped_d": 0} if d else {})


def test_accumulate_equals_full_batch(tmp_path):
    big = batches(1, b=2 * B)[0]
    halves = [{k: v[i * B : (i + 1) * B] for k, v in big.items()} for i in range(2)]
    a, b = (
        build(make_opt(tmp_path, accumulate=1)),
        build(make_opt(tmp_path, accumulate=2)),
    )
    run(a, [big])
    run(b, halves)
    for x, y in zip(a.net_g.parameters(), b.net_g.parameters(), strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=1e-6)


def test_no_sync_on_non_boundary_micro_batches(tmp_path):
    m = build(make_opt(tmp_path, d=True, accumulate=2))
    entered = []

    class Spy(nn.Module):
        def __init__(self, net, tag) -> None:
            super().__init__()
            self.net, self.tag = net, tag

        def forward(self, x):
            return self.net(x)

        @contextmanager
        def no_sync(self):
            entered.append((self.tag, m.micro))
            yield

    m.net_g_fwd, m.net_d_fwd = Spy(m.net_g, "g"), Spy(m.net_d, "d")
    run(m, batches(4))
    assert entered == [("g", 0), ("d", 0), ("g", 2), ("d", 2)]


def test_nan_gradient_is_skipped_cleanly(tmp_path):
    m = build(make_opt(tmp_path, ema=0.9))
    data = batches(3)
    run(m, data[:1])  # first update: EMA initialised
    m.get_current_log()
    live, avg, state = clone(m.net_g), clone(m.ema.module), opt_tensors(m.g.opt)
    hook = next(m.net_g.parameters()).register_hook(
        lambda g: torch.full_like(g, float("nan"))
    )
    assert run(m, data[1:2]) == [True]  # the window closes, the step is skipped
    hook.remove()
    assert same(live, clone(m.net_g))
    assert same(avg, clone(m.ema.module))
    assert same(state, opt_tensors(m.g.opt))
    assert all(p.grad is None for p in m.net_g.parameters())
    log = m.get_current_log()
    assert log["skipped_g"] == 1
    assert "grad_norm_g" not in log  # mean over successful updates: none here
    assert m.update == 2
    run(m, data[2:])  # training continues
    log = m.get_current_log()
    assert log["skipped_g"] == 0
    assert "grad_norm_g" in log
    assert not same(live, clone(m.net_g))


def test_nan_objective_with_finite_grads_raises_at_log(tmp_path):
    class NaNTerm:
        def active(self, _it):
            return True

        def compute(self, _ctx):
            return torch.tensor(float("nan"))  # detached: the gradients stay finite

    m = build(make_opt(tmp_path))
    m.g.terms.append(NaNTerm())
    g0 = clone(m.net_g)
    run(m, batches(1))
    assert m.g.skipped == 0  # finite grads: the update happened
    assert not same(g0, clone(m.net_g))
    with pytest.raises(FloatingPointError, match="l_g_total"):
        m.check_finite()
    with pytest.raises(FloatingPointError, match="l_g_total"):
        m.get_current_log()


def test_check_finite_catches_bad_weights(tmp_path):
    m = build(make_opt(tmp_path, d=True))
    run(m, batches(1))
    m.get_current_log()
    with torch.no_grad():
        next(m.net_d.buffers()).fill_(float("inf"))  # SN u vector
    with pytest.raises(FloatingPointError, match="non-finite weights"):
        m.check_finite()


# ---------------------------------------------------------------------------- (ii) (iii) eval_weights
def test_sf_export_and_validation_restore_exactly(tmp_path):
    m = build(make_opt(tmp_path, optim=SF))
    run(m, batches(3))
    net, opt = m.net_g, m.g.opt
    assert m.g.sf
    list(net.modules())[2].training = False  # non-uniform module modes must survive
    live, modes = clone(net), [x.training for x in net.modules()]
    flags = [g["train_mode"] for g in opt.param_groups]
    opt.eval()  # the oracle: x, then an exact manual restore
    x = {k: v.clone() for k, v in net.state_dict().items()}
    with torch.no_grad():
        for p, v in zip(net.parameters(), live, strict=False):
            p.copy_(v)
    for g, f in zip(opt.param_groups, flags, strict=True):
        g["train_mode"] = f
    assert not sd_equal(x, net.state_dict())  # the swap is visible

    def unchanged():
        assert same(live, clone(net))
        assert [g["train_mode"] for g in opt.param_groups] == flags == [True]
        assert [y.training for y in net.modules()] == modes

    assert sd_equal(m.export_state_dicts()["net_g"], x)
    unchanged()
    seen = []
    m.metrics = {
        "probe": lambda *_: (
            seen.append(sd_equal(net.state_dict(), x) and not net.training) or 0.0
        )
    }
    m.opt["val"]["metrics"] = {"probe": {"better": "lower"}}
    m.validation(loader(), 3, None, save_img=False)
    assert seen == [True, True]
    unchanged()
    assert m.best_metric_results["valset"]["probe"] == {
        "better": "lower",
        "val": 0.0,
        "iter": 3,
    }


def test_sf_disabled_exports_live_weights(tmp_path):
    m = build(make_opt(tmp_path, optim=SF | {"schedule_free": False}))
    run(m, batches(3))
    assert not m.g.sf
    assert sd_equal(m.export_state_dicts()["net_g"], m.net_g.state_dict())


def test_ema_export_is_the_average(tmp_path):
    m = build(make_opt(tmp_path, d=True, ema=0.9, optim=SF))
    run(m, batches(2))
    out = m.export_state_dicts()
    assert sd_equal(out["net_g"], m.ema.module.state_dict())
    assert sd_equal(out["net_d"], m.net_d.state_dict())
    assert all(
        v.device.type == "cpu" and v.is_contiguous()
        for sd in out.values()
        for v in sd.values()
    )


# ---------------------------------------------------------------------------- (iv) state format, save
def test_state_format_and_exports(tmp_path):
    opt = make_opt(tmp_path, d=True, ema=0.9, optim=SF)
    opt["train"]["scheduler"] = {"type": "MultiStepLR", "milestones": [2], "gamma": 0.5}
    opt["logger"]["keep_states"] = 2
    m = build(opt)
    m.validation(loader(), 0, None, save_img=False)
    for _ in range(3):
        run(m, batches(1))
        m.save()
    m.save(final=True)
    states = tmp_path / "training_states"
    kept = sorted(p.name for p in states.iterdir())
    assert kept == ["2.state", "3.state"]  # keep_states = 2
    names = {p.name for p in (tmp_path / "models").iterdir()}
    assert names == {f"net_{n}_{u}.pth" for n in "gd" for u in (1, 2, 3, "latest")}
    assert not list(tmp_path.rglob("*.tmp"))
    st = torch.load(states / "3.state", map_location="cpu", weights_only=True)
    assert set(st) == STATE_KEYS
    assert (st["format"], st["update"]) == (2, 3)
    assert set(st["nets"]) == set(st["phases"]) == {"net_g", "net_d"}
    assert sd_equal(st["nets"]["net_g"], m.net_g.state_dict())
    assert sd_equal(st["ema"], m.ema.module.state_dict())
    ph = st["phases"]["net_g"]
    assert set(ph) == {"optimizer", "scaler", "schedulers", "clip_ema"}
    assert ph["scaler"] == {}
    assert ph["clip_ema"] is None  # grad_clip = "spike" only
    assert set(ph["optimizer"]) == {"type", "version", "sf", "names", "state"}
    assert (ph["optimizer"]["type"], ph["optimizer"]["sf"]) == ("adan_sf", True)
    assert ph["optimizer"]["names"] == [n for n, _ in m.net_g.named_parameters()]
    assert [s["type"] for s in ph["schedulers"]] == ["MultiStepLR"]
    assert m.current_lr() == [pytest.approx(5e-3)]
    assert set(st["rng"]) == {"python", "numpy", "neosr", "torch", "cuda", "loader"}
    tr = opt["train"]
    assert st["run"] == {
        "network_g": opt["network_g"], "network_d": opt["network_d"], "use_amp": False,
        "bfloat16": False, "ema": 0.9, "world_size": 1, "optim_g": tr["optim_g"],
        "optim_d": tr["optim_d"], "scheduler": tr["scheduler"],
        "losses": {"pixel_opt": tr["pixel_opt"], "gan_opt": tr["gan_opt"]},
        "train": {"clamp": True, "grad_clip": True, "ema_power": None} | dict.fromkeys(QAT_KEYS),
        "data": {"accumulate": 1, "batch_size": B, "patch_size": P},
    }  # fmt: skip
    assert st["best"] == m.best_metric_results
    assert set(st["best"]["valset"]) == {"psnr"}
    exp = torch.load(tmp_path / "models" / "net_g_latest.pth", weights_only=True)
    assert set(exp) == {"params"}
    assert sd_equal(exp["params"], m.ema.module.state_dict())


def test_save_only_at_a_boundary_and_outside_eval(tmp_path):
    m = build(make_opt(tmp_path, accumulate=2))
    run(m, batches(1))
    with pytest.raises(RuntimeError, match="boundary"):
        m.save()
    run(m, batches(1))
    with m.eval_weights(), pytest.raises(RuntimeError, match="eval_weights"):
        m.save()
    m.save()
    assert (tmp_path / "training_states" / "1.state").is_file()


# ---------------------------------------------------------------------------- (v) resume
def _resume_opt(tmp, path, **kw):
    opt = make_opt(tmp, **kw)
    opt["path"]["resume_state"] = path
    return opt


@pytest.mark.parametrize(("saved_d", "now_d"), [(False, True), (True, False)])
def test_resume_with_changed_nets_raises(tmp_path, saved_d, now_d):
    m = build(make_opt(tmp_path, d=saved_d))
    run(m, batches(1))
    m.save()
    with pytest.raises(ConfigError, match=r"nets .* saved, .* configured"):
        build(_resume_opt(tmp_path, tmp_path / "training_states" / "1.state", d=now_d))


def test_resume_refuses_legacy_and_unreadable_states(tmp_path):
    torch.save(
        {"iter": 5, "epoch": 1, "optimizers": [], "schedulers": []},
        tmp_path / "5.state",
    )
    with pytest.raises(ConfigError, match=r"net_g_5\.pth"):
        build(_resume_opt(tmp_path, tmp_path / "5.state"))
    (tmp_path / "6.state").write_bytes(b"not a checkpoint")
    with pytest.raises(RuntimeError, match=r"6\.state"):
        build(_resume_opt(tmp_path, tmp_path / "6.state"))


def test_resume_with_a_changed_network_shape_raises(tmp_path):
    m = build(make_opt(tmp_path))
    run(m, batches(1))
    m.save()
    opt = _resume_opt(tmp_path, tmp_path / "training_states" / "1.state")
    opt["network_g"]["num_feat"] = 4
    with pytest.raises(ConfigError, match="net_g does not match"):
        build(opt)


def test_resume_continues_exactly(tmp_path):
    kw = {"d": True, "ema": 0.9, "optim": SF}
    data = batches(4)
    a = build(make_opt(tmp_path / "a", **kw))
    run(a, data)

    b = build(make_opt(tmp_path / "b", **kw))
    b.epoch = 7
    run(b, data[:2])
    b.validation(loader(), 2, None, save_img=False)
    b.save()
    expected_draws = torch.rand(
        3
    )  # what the training stream yields right after the save
    r = build(
        _resume_opt(
            tmp_path / "b", tmp_path / "b" / "training_states" / "2.state", **kw
        ),
        seed=5,
    )
    assert (r.update, r.epoch, r.micro) == (2, 7, 0)
    assert r.best_metric_results == b.best_metric_results
    r.restore_rng()
    assert torch.equal(torch.rand(3), expected_draws)
    run(r, data[2:])
    for x, y in ((a.net_g, r.net_g), (a.net_d, r.net_d), (a.ema.module, r.ema.module)):
        assert same(clone(x), clone(y))
    for n in ("net_g", "net_d"):
        assert same(opt_tensors(a.phases[n].opt), opt_tensors(r.phases[n].opt))


def test_resume_enabling_ema_starts_from_the_resumed_weights(tmp_path):
    m = build(make_opt(tmp_path))
    run(m, batches(1))
    m.save()
    r = build(
        _resume_opt(tmp_path, tmp_path / "training_states" / "1.state", ema=0.9), seed=3
    )
    assert sd_equal(r.ema.module.state_dict(), m.net_g.state_dict())
    run(r, batches(1, seed=1))  # first update after resume copies the live weights
    assert same(clone(r.ema.module), clone(r.net_g))


# ---------------------------------------------------------------------------- otf, config errors
def test_otf_feed_data_synthesises_the_pair(tmp_path):
    opt = make_opt(tmp_path, model_type="otf")
    opt["datasets"]["train"] |= {
        "queue_size": 4, "jpeg_compress": False, "resize_prob": [0.3, 0.4, 0.3], "resize_range": [0.5, 1.5],
        "gaussian_noise_prob": 0.5, "noise_range": [0, 2], "poisson_scale_range": [0.05, 0.25],
        "gray_noise_prob": 0.3, "second_blur_prob": 0.5, "resize_prob2": [0.3, 0.4, 0.3],
        "resize_range2": [0.3, 1.5], "gaussian_noise_prob2": 0.5, "noise_range2": [0, 2],
        "poisson_scale_range2": [0.05, 0.1], "gray_noise_prob2": 0.3,
    }  # fmt: skip
    m = build(opt)
    g = torch.Generator().manual_seed(0)

    def kernel(k):
        w = torch.rand(B, k, k, generator=g)
        return w / w.sum((1, 2), keepdim=True)

    for _ in range(3):  # fills the pool, then swaps
        m.feed_data({
            "gt": torch.rand(B, 3, 32, 32, generator=g),
            "kernel1": kernel(7),
            "kernel2": kernel(9),
            "sinc_kernel": kernel(21),
        })
        assert m.lq.shape == (B, 3, P, P)
        assert m.gt.shape == (B, 3, P * S, P * S)
        assert not m.lq.requires_grad
        m.optimize_parameters()
    assert m.update == 3


@pytest.mark.parametrize(
    ("optim", "match"),
    [
        ({"type": "adamww", "lr": 1e-3}, "did you mean 'adamw'"),
        ({"type": "AdamW", "lrr": 1e-3}, "lrr"),
    ],
)
def test_optimizer_config_errors(tmp_path, optim, match):
    with pytest.raises(ConfigError, match=match):
        build(make_opt(tmp_path, optim=optim))


# ---------------------------------------------------------------------------- DDP seam, CUDA
def test_ddp_single_rank_gloo(tmp_path):
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'pg'}", rank=0, world_size=1
    )
    try:
        m = build(
            make_opt(tmp_path, d=True, accumulate=2),
            rt=Runtime(torch.device("cpu"), 0, 1, 0, True),
        )
        assert isinstance(m.net_g_fwd, DistributedDataParallel)
        assert isinstance(m.net_d_fwd, DistributedDataParallel)
        assert m.net_g_fwd.broadcast_buffers
        assert not m.net_d_fwd.broadcast_buffers
        assert m.net_d_adv is m.net_d  # G's adversarial pass bypasses D's reducer
        run(m, batches(4))
        assert set(m.get_current_log()) == G_KEYS | D_KEYS
        m.check_finite()
        m.validation(loader(), 2, None, save_img=False)
        m.save()
        st = torch.load(tmp_path / "training_states" / "2.state", weights_only=True)
        assert not any(k.startswith("module.") for k in st["nets"]["net_g"])
    finally:
        dist.destroy_process_group()


def _ddp_gan_worker(rank: int, tmp: str, init: str, out: str) -> None:
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=2)
    try:
        # seed + rank (T3 D15): SN u/v and weights start different on each rank
        rt = Runtime(torch.device("cpu"), rank, 2, 0, True)
        m = build(make_opt(Path(tmp), d=True), rt=rt, seed=rank)
        res = {"init": clone(m.net_d)}
        run(m, batches(2, seed=rank))  # rank-specific data
        res["trained"] = clone(m.net_g) + clone(m.net_d)
        m.get_current_log()  # the meter is empty: check_finite reduces only the flag
        if rank == 1:
            with torch.no_grad():
                next(m.net_d.buffers()).fill_(float("nan"))
        try:
            m.check_finite()
            res["raised"] = False
        except FloatingPointError:
            res["raised"] = True
        torch.save(res, f"{out}.{rank}")
    finally:
        dist.destroy_process_group()


def test_ddp_two_rank_gan_syncs_d_buffers_and_check_finite(tmp_path):
    out = str(tmp_path / "res")
    torch.multiprocessing.spawn(
        _ddp_gan_worker,
        args=(str(tmp_path), f"file://{tmp_path / 'pg'}", out),
        nprocs=2,
        join=True,
    )
    r0, r1 = (torch.load(f"{out}.{r}") for r in (0, 1))
    assert same(r0["init"], r1["init"])  # D's buffers start equal (broadcast once)
    assert same(r0["trained"], r1["trained"])  # and they evolve identically
    assert r1["raised"]
    assert r0["raised"]  # a bad buffer on rank 1 stops rank 0 too


def test_enable_qat_is_rejected_until_supported(tmp_path):
    opt = make_opt(tmp_path)
    opt["train"]["enable_qat"] = True
    with pytest.raises(ConfigError, match="enable_qat"):
        build(opt)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_bf16_gan_channels_last(tmp_path):
    opt = make_opt(tmp_path, d=True, ema=0.9, optim=SF) | {
        "use_amp": True,
        "bfloat16": True,
    }
    m = build(opt, rt=Runtime(torch.device("cuda", 0), 0, 1, 0, False))
    data = [{k: v.cuda() for k, v in x.items()} for x in batches(3)]
    run(m, data)
    assert m.lq.is_contiguous(memory_format=torch.channels_last)
    assert m.net_g.body[0].weight.is_contiguous(memory_format=torch.channels_last)
    log = m.get_current_log()
    assert set(log) == G_KEYS | D_KEYS
    assert log["skipped_g"] == log["skipped_d"] == 0
    m.check_finite()
    m.validation(loader(), 3, None, save_img=False)
    m.save(final=True)
    exp = torch.load(Path(tmp_path) / "models" / "net_g_3.pth", weights_only=True)[
        "params"
    ]
    assert all(v.device.type == "cpu" for v in exp.values())
