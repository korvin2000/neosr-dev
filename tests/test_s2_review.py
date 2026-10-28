"""Stage 2 review fixes (two Codex reviews of the merged branch): one regression test per finding."""

import importlib.util
import logging
import math
import weakref
from functools import partial
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from neosr.archs.arch_util import DySample, window_sdpa
from neosr.cli import convert_args
from neosr.config import ConfigError, load_config, validate_train
from neosr.losses.base import restoring_clamp
from neosr.losses.consistency_loss import consistency_loss
from neosr.metrics.calculate import (
    _metric_net,
    _metric_nets,
    _ssim_kernel,
    calculate_ssim_pt,
)
from neosr.optimizers import adamw_sf, adan, adan_sf
from neosr.optimizers.adan import clip_grad_norm_
from neosr.utils import get_root_logger
from tests.test_config import BASE, HEAD, write
from tests.test_model import (
    batches,
    build,
    clone,
    loader,
    make_opt,
    opt_tensors,
    run,
    same,
)
from tests.test_s2_trainer import Log, _phase, _update


# ---------------------------------------------------------------------------- 1 consistency clamp
def test_restoring_clamp_with_bounds_gates_like_the_unit_clamp():
    x = torch.tensor([-0.5, -0.5, 0.0, 0.3, 0.3, 1.0, 1.5, 1.5], requires_grad=True)
    g = torch.tensor([1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0])
    lo, hi = 1 / 255, 0.9
    y = restoring_clamp(x, lo, hi)
    assert torch.equal(y, x.detach().clamp(lo, hi))
    y.backward(g)
    assert x.grad.tolist() == [0.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 0.0]


def test_consistency_loss_passes_inward_gradients_of_out_of_range_outputs():
    torch.manual_seed(0)
    gt = torch.rand(1, 3, 32, 32) * 0.8 + 0.1
    raw = gt.clone()
    raw[..., :4, :4] = -0.3  # below range: descent must raise these
    raw[..., -4:, -4:] = 1.4  # above range: descent must lower these
    raw.requires_grad_()
    loss = consistency_loss(blur=False, cosim=False)
    loss(restoring_clamp(raw), gt).backward()
    g = raw.grad
    assert (g[..., :4, :4] <= 0).all()
    assert g[..., :4, :4].any()
    assert (g[..., -4:, -4:] >= 0).all()
    assert g[..., -4:, -4:].any()
    # in-range values: unchanged from the legacy hard clamp
    ref = gt.clone().requires_grad_()
    loss(ref.clamp(1 / 255, 1), gt.flip(-1)).backward()
    new = gt.clone().requires_grad_()
    loss(new, gt.flip(-1)).backward()
    assert torch.equal(ref.grad, new.grad)


# ---------------------------------------------------------------------------- 3 adan max_grad_norm
@pytest.mark.parametrize("make", [adan, adan_sf], ids=["adan", "adan_sf"])
def test_adan_clips_by_the_group_max_grad_norm(make):
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(16))
    q = torch.nn.Parameter(p.detach().clone())
    resumed = make([p], lr=1e-2, max_grad_norm=0.0)
    resumed.param_groups[0]["max_grad_norm"] = (
        0.5  # what a resume restores over the config
    )
    reference = make([q], lr=1e-2, max_grad_norm=0.5)
    for i in range(3):
        g = torch.randn(16, generator=torch.Generator().manual_seed(i)) * 10
        p.grad, q.grad = g.clone(), g.clone()
        resumed.step()
        reference.step()
    assert torch.equal(p, q)
    for k, v in reference.state[q].items():
        assert torch.equal(resumed.state[p][k], v), k


def test_adan_clip_norm_spans_groups_and_skips_unclipped_groups():
    a, b = torch.nn.Parameter(torch.zeros(4)), torch.nn.Parameter(torch.zeros(4))
    opt = adan([{"params": [a], "max_grad_norm": 1.0}, {"params": [b]}], lr=0.0)
    a.grad, b.grad = torch.full((4,), 3.0), torch.full((4,), 4.0)
    clip_grad_norm_(opt.param_groups)
    norm = torch.tensor(10.0)  # sqrt(4 * 9 + 4 * 16): both groups count
    torch.testing.assert_close(a.grad, torch.full((4,), 3.0) / (norm + 1e-8))
    assert torch.equal(b.grad, torch.full((4,), 4.0))


# ---------------------------------------------------------------------------- 10 adamw_sf version
@pytest.mark.parametrize("foreach", [True, False])
def test_adamw_sf_step_bumps_the_parameter_version(foreach):
    p = torch.nn.Parameter(torch.ones(4))
    opt = adamw_sf([p], lr=1e-2, foreach=foreach)
    opt.train()
    p.grad = torch.ones(4)
    v = p._version
    opt.step()
    assert p._version > v
    v = p._version
    opt.eval()
    assert p._version > v


# ---------------------------------------------------------------------------- 4 run record
def test_resume_warns_about_global_objective_keys(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(
        get_root_logger(), "propagate", True
    )  # caplog listens on the root logger
    m = build(make_opt(tmp_path))
    run(m, batches(1))
    m.save()
    opt = make_opt(tmp_path)
    opt["path"]["resume_state"] = tmp_path / "training_states" / "1.state"
    opt["train"]["match_lq_colors"] = True
    with caplog.at_level(logging.WARNING, logger="neosr"):
        build(opt)
    assert "train.match_lq_colors: None -> True" in caplog.text
    assert "wavelet" not in caplog.text  # unset before and after: not a change


def test_resume_of_a_state_older_than_a_train_key_does_not_warn(
    tmp_path, caplog, monkeypatch
):
    monkeypatch.setattr(get_root_logger(), "propagate", True)
    opt = make_opt(tmp_path)
    opt["train"]["match_lq_colors"] = True
    m = build(opt)
    run(m, batches(1))
    m.save()
    path = tmp_path / "training_states" / "1.state"
    state = torch.load(path, map_location="cpu", weights_only=True)
    del state["run"]["train"]["match_lq_colors"]  # saved before the key was recorded
    del state["run"]["network_g"]["num_conv"]  # a config sub-key set since: still a change
    torch.save(state, path)
    opt = make_opt(tmp_path)
    opt["path"]["resume_state"] = path
    opt["train"]["match_lq_colors"] = True
    with caplog.at_level(logging.WARNING, logger="neosr"):
        build(opt)
    assert "match_lq_colors" not in caplog.text
    assert "network_g.num_conv: None -> 2" in caplog.text


# ---------------------------------------------------------------------------- 5 mean/std
def test_validate_train_rejects_identity_normalisation_like_the_datasets(tmp_path):
    body = BASE.replace(
        "batch_size = 2", "batch_size = 2\nmean = [0, 0, 0]\nstd = [1, 1, 1]", 1
    )
    with pytest.raises(
        ConfigError, match=r"\[datasets.train\] 'mean' is not supported"
    ):
        validate_train(
            load_config(write(tmp_path, body, HEAD), is_train=True, root=tmp_path)
        )


# ---------------------------------------------------------------------------- 6 skip streak
def test_skip_streak_survives_a_resume():
    _, _, a = _phase(
        True, names=["0.weight", "0.bias", "1.weight", "1.bias"], max_skip_streak=3
    )
    assert _update(a, math.nan) is False
    assert _update(a, math.nan) is False
    sd = a.state_dict()
    assert sd["streak"] == 2
    _, _, b = _phase(
        True, names=["0.weight", "0.bias", "1.weight", "1.bias"], max_skip_streak=3
    )
    b.load_state_dict(sd, Log())
    with pytest.raises(FloatingPointError, match="3 consecutive"):
        _update(b, math.nan)
    del sd["streak"]  # a state saved before the key existed
    b.load_state_dict(sd, Log())
    assert b.streak == 0


# ---------------------------------------------------------------------------- 9 SSIM channel scope
class LivePeak(TorchDispatchMode):
    """Peak bytes of the op outputs alive at once (freed when their last reference goes)."""

    def __init__(self) -> None:
        super().__init__()
        self.live = self.peak = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in out if isinstance(out, tuple | list) else (out,):
            if isinstance(t, torch.Tensor) and not t._is_view():
                n = t.untyped_storage().nbytes()
                self.live += n
                weakref.finalize(t, self._free, n)
        self.peak = max(self.peak, self.live)
        return out

    def _free(self, n: int) -> None:
        self.live -= n


def _ssim_loop_before_the_fix(img, img2):
    """calculate_ssim_pt's channel loop as merged (crop 0, no Y): the reference score and memory."""
    n, c, h, w = img.shape
    k, c1, c2 = _ssim_kernel(img.device), (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssims = []
    for i in range(c):
        x, y = img[:, i].double(), img2[:, i].double()
        stats = torch.stack((x, y, x * x, y * y, x * y), 1).view(n * 5, 1, h, w)
        stats = F.conv2d(F.conv2d(stats, k), k.view(1, 1, 1, 11))
        mu1, mu2, xx, yy, xy = stats.view(n, 5, h - 10, w - 10).unbind(1)
        mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * (xy - mu1_mu2) + c2)) / (
            (mu1_sq + mu2_sq + c1) * ((xx - mu1_sq) + (yy - mu2_sq) + c2)
        )
        ssims.append(ssim_map.mean((-2, -1)))
    return torch.stack(ssims, -1).mean()


def test_ssim_frees_each_channel_before_the_next_with_an_identical_score():
    g = torch.Generator().manual_seed(0)
    a = torch.randint(0, 256, (1, 3, 96, 96), generator=g).float()
    b = (a + torch.randint(-20, 21, a.shape, generator=g)).clamp(0, 255)
    with LivePeak() as old:
        ref = _ssim_loop_before_the_fix(a, b)
    with LivePeak() as new:
        got = calculate_ssim_pt(a, b, crop_border=0)
    assert torch.equal(got, ref)
    channel_stats = 5 * 96 * 96 * 8  # one channel's fp64 statistics
    assert (
        old.peak - new.peak > channel_stats
    )  # the previous channel's maps no longer overlap


# ---------------------------------------------------------------------------- 7 window_sdpa trace
def _convert_py():
    spec = importlib.util.spec_from_file_location(
        "convert_py", Path(__file__).resolve().parents[1] / "convert.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


convert_py = _convert_py()


def _convert_argv(tmp, *extra):
    argv = ["--input", str(tmp / "w.pth"), "-net", "compact", "-s", "2"]
    return convert_args([*argv, "--output", str(tmp / "m.onnx"), *extra])


@pytest.fixture
def onnx_deps():
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")


class PaddedWindows(nn.Module):
    """Pad to the window, masked window attention, crop: one window at the 20x20 trace size."""

    def __init__(self, ws: int = 20, c: int = 4) -> None:
        super().__init__()
        self.ws, n = ws, ws * ws
        self.inp, self.out = (
            nn.Conv2d(3, c, 3, padding=1),
            nn.Conv2d(c, 3, 3, padding=1),
        )
        self.bias = nn.Parameter(torch.randn(1, n, n))
        self.register_buffer("tri", torch.triu(torch.ones(n, n), 1) * -100.0)

    def forward(self, x):
        b, _, h, w = x.shape
        ws, x = self.ws, self.inp(x)
        x = F.pad(x, (0, -w % ws, 0, -h % ws))
        c, hp, wp = x.shape[1:]
        t = x.view(b, c, hp // ws, ws, wp // ws, ws).permute(0, 2, 4, 3, 5, 1)
        t = t.reshape(-1, 1, ws * ws, c)
        nw = (hp // ws) * (wp // ws)
        mask = (torch.arange(nw) > 0).float().view(
            -1, 1, 1
        ) * self.tri  # shift-mask-like
        t = window_sdpa(t, t, t, self.bias, mask, scale=0.5)
        t = t.reshape(b, hp // ws, wp // ws, ws, ws, c).permute(0, 5, 1, 3, 2, 4)
        return self.out(t.reshape(b, c, hp, wp)[..., :h, :w])


def test_dynamic_export_traced_at_one_window_runs_at_several(tmp_path, onnx_deps):
    torch.manual_seed(0)
    net = PaddedWindows().eval()
    assert convert_py.TRACE[-2:] == (
        20,
        20,
    )  # one window at the trace shape, four at 32x24
    convert_py.to_onnx(
        _convert_argv(tmp_path), net
    )  # verified at 20x20, 32x24 and batch 2


# ---------------------------------------------------------------------------- 8 fp16 DySample
def test_module_path_of_exported_node_names():
    mp = convert_py._module_path
    assert mp("/a/a.0/a.0.0/up/offset/Conv") == "a.0.0.up.offset"
    assert mp("/ups.0/ups.0.1/Mul_3") == "ups.0.1"
    assert mp("/Mul") == mp("Constant_7") == ""


def test_fp16_export_keeps_dysample_in_fp32_at_a_wide_input(tmp_path, onnx_deps):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(3, 4, 1), nn.Sequential(DySample(4, 3, 2, 4))).eval()
    args = _convert_argv(
        tmp_path, "-fp16", "-static", "3", "4", "1024"
    )  # output width 2048
    convert_py.to_onnx(
        args, net
    )  # verified: fp16 ORT vs fp32 torch, max 2e-2, mean 2e-3
    import onnx  # ruff: ignore[import-outside-top-level]

    graph = onnx.load(str(tmp_path / "m_fp16.onnx")).graph
    assert graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    grid = [n for n in graph.node if n.op_type == "GridSample"]
    assert grid
    assert all(n.name.startswith("/1/1.0/") for n in grid)


# ---------------------------------------------------------------------------- 2 learned metric OOM
def test_metric_net_reunites_a_net_split_by_a_failed_move(monkeypatch):
    net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    net[1].to("meta")  # as if the move to the device ran out of memory after net[0]
    moves = []
    monkeypatch.setattr(net, "to", lambda d: moves.append(d) or net)
    monkeypatch.setitem(_metric_nets, "split", net)
    assert _metric_net("split", torch.device("cpu")) is net
    assert moves == [torch.device("cpu")]


def test_learned_metric_oom_falls_back_to_the_cpu_per_image(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(get_root_logger(), "propagate", True)
    calls, emptied = [], []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(1))

    def learned(img, img2, device):  # DISTS/TOPIQ's signature: bound to the run device
        calls.append(torch.device(device).type)
        if torch.device(device).type != "cpu":
            raise torch.OutOfMemoryError("CUDA out of memory (test)")
        return (img - img2).abs().mean()

    opt = make_opt(tmp_path, is_train=False)
    opt["val"]["metrics"]["learned"] = {"type": "calculate_dists", "better": "lower"}
    m = build(opt)
    m.metrics = {
        "psnr": m.metrics["psnr"],
        "learned": partial(learned, device=torch.device("cuda")),
    }
    logged = []
    m.log_metrics = lambda name, results, *a: logged.append(results)
    with caplog.at_level(logging.WARNING, logger="neosr"):
        m.validation(loader(n=2), "x", None, False)
    assert calls == ["cuda", "cpu", "cuda", "cpu"]  # each image tries the device first
    assert len(emptied) == 2
    warns = [r for r in caplog.records if "learned validation metric" in r.getMessage()]
    assert len(warns) == 1  # once per validation event
    assert logged[0]["learned"] > 0
    # a non-learned metric's OOM still propagates
    m.metrics = {
        "psnr": lambda q, gt: (_ for _ in ()).throw(torch.OutOfMemoryError("x"))
    }
    with pytest.raises(torch.OutOfMemoryError):
        m.validation(loader(n=1), "x", None, False)


# ---------------------------------------------------------------------------- 11 GAN boundary resume
START, UPDATES = 3, 6  # the GAN term and D's update join at update 4


def _gan_opt(tmp):
    opt = make_opt(tmp, d=True, accumulate=2)
    opt["train"]["grad_clip"] = "spike"
    opt["train"]["gan_opt"]["start_iter"] = START
    return opt


def _updates(m, data, n):
    """n updates (accumulate 2: two micro-batches each); the window log after each."""
    logs = []
    for _ in range(n):
        run(m, data[2 * m.update : 2 * m.update + 2])
        logs.append(m.get_current_log())
    return logs


@pytest.mark.parametrize("at", [START - 1, START])
def test_resume_is_exact_across_the_gan_activation(tmp_path, at):
    data = batches(2 * UPDATES)
    a = build(_gan_opt(tmp_path / "a"))
    logs_a = _updates(a, data, UPDATES)
    assert (
        "l_g_gan" not in logs_a[START - 1] and "l_g_gan" in logs_a[START]
    )  # the boundary

    b = build(_gan_opt(tmp_path / "b"))
    _updates(b, data, at)
    b.save()
    opt = _gan_opt(tmp_path / "b")
    opt["path"]["resume_state"] = tmp_path / "b" / "training_states" / f"{at}.state"
    r = build(opt, seed=5)
    r.restore_rng()
    assert r.update == at
    logs_r = _updates(r, data, UPDATES - at)

    assert logs_r == logs_a[at:]
    for x, y in ((a.net_g, r.net_g), (a.net_d, r.net_d)):
        assert same(clone(x), clone(y))
    for n in ("net_g", "net_d"):
        assert same(opt_tensors(a.phases[n].opt), opt_tensors(r.phases[n].opt))
        pa, pr = a.phases[n], r.phases[n]
        assert torch.equal(pa.norm_ema, pr.norm_ema)  # the spike clip's EMA
