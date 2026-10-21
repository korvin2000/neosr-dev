"""AetherNet registration and QAT end to end (design T5, adapted to the new trainer), on the CPU."""

import argparse
from collections import Counter
from itertools import starmap
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.ao.nn.qat as nnqat
import torch.ao.quantization as tq
from torch.utils.data import DataLoader, Dataset

from neosr.archs import build_network
from neosr.archs.aether_arch import DeploymentNorm, aether, aether_small
from neosr.config import ConfigError, validate_train
from neosr.models import build_model
from neosr.models.checkpoint import read_weights
from neosr.models.image import observers_frozen
from neosr.runtime import Runtime

CPU = Runtime(torch.device("cpu"), 0, 1, 0, False)
S, P, B = 2, 8, 2  # scale, lq patch, batch


def net_small(**kw) -> aether:
    torch.manual_seed(0)
    return build_network({"type": "aether_small", **kw}, scale=S)


def fqs(net):
    return [m for m in net.modules() if isinstance(m, tq.FakeQuantize)]


def observed(net, n=2):
    net.set_qat_phase(fake_quant=True, observe=True, freeze_norm=False)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for _ in range(n):
            net(torch.rand(B, 3, 12, 12, generator=g))


def tensors(net):
    return [t.detach().clone() for t in (*net.parameters(), *net.buffers())]


def same(a, b):
    return len(a) == len(b) and all(starmap(torch.equal, zip(a, b, strict=True)))


def norm_stats(net):
    return [
        t.clone()
        for m in net.modules()
        if isinstance(m, DeploymentNorm)
        for t in (m.running_mean, m.running_var)
    ]


def observer_buffers(net):
    return [
        t.clone()
        for m in fqs(net)
        for t in (
            m.scale,
            m.activation_post_process.min_val,
            m.activation_post_process.max_val,
        )
    ]


# ------------------------------------------------------------------ arch
@pytest.mark.parametrize("name", ["aether_small", "aether_medium", "aether_large"])
@pytest.mark.parametrize("scale", [2, 3, 4])
def test_variants_build_at_the_run_scale(name, scale):
    net = build_network({"type": name}, scale=scale).eval()
    with torch.no_grad():
        assert net(torch.rand(1, 3, 6, 5)).shape == (1, 3, 6 * scale, 5 * scale)


def test_bad_scale_and_fixed_variant_kwargs_raise():
    with pytest.raises(ValueError, match="scale=1"):
        build_network({"type": "aether_small"}, scale=1)
    with pytest.raises(TypeError, match="embed_dim"):
        aether_small(scale=2, embed_dim=64)
    with pytest.raises(ValueError, match="divisible"):
        aether(embed_dim=90, depths=(4, 4, 4, 4), scale=2)


def test_prepare_qat_topology_and_idle_state():
    net = net_small()
    stats = norm_stats(net)
    net.prepare_qat()
    assert len(fqs(net)) == 226  # T5 §0
    assert sum(isinstance(m, nnqat.Conv2d) for m in net.modules()) == 71
    for name in ("conv_first", "conv_before_upsample.0", "conv_last"):  # T5 D8
        m = net.get_submodule(name)
        assert type(m) is torch.nn.Conv2d
        assert not hasattr(m, "activation_post_process")
    assert same(stats, norm_stats(net))  # priming ran in eval (T5 D10)
    for m in fqs(net):  # idle until the caller sets a phase
        assert (m.fake_quant_enabled.item(), m.observer_enabled.item()) == (0, 0)
        obs = m.activation_post_process
        if m.is_per_channel:  # materialised: EMA / DDP copy buffers in place
            assert m.scale.numel() == obs.min_val.numel() > 1
        else:  # the dummy activation range is forgotten
            assert obs.min_val.item() == float("inf")
    with pytest.raises(RuntimeError, match="already"):
        net.prepare_qat()


@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("name", ["aether_small", "aether_medium"])
def test_one_observed_forward_leaves_every_buffer_finite(name, residual):
    """check_finite() scans all float buffers: no observer may stay at its +-inf reset."""
    net = build_network({"type": name}, scale=S)
    net.prepare_qat(quantize_residual=residual)
    net.train()
    observed(net, n=1)
    bad = [
        k
        for k, v in net.state_dict().items()
        if v.is_floating_point() and not v.isfinite().all()
    ]
    assert bad == []
    has_res = any(
        ".residual_quant.activation_post_process." in k for k in net.state_dict()
    )
    assert has_res is residual
    assert aether.qat_state(net.state_dict()) == {"quantize_residual": residual}


def test_qat_state_detects_only_qat_files():
    assert aether.qat_state(net_small().state_dict()) is None


def test_set_qat_phase_flags_and_norm_freeze():
    net = net_small()
    net.prepare_qat()
    net.train()
    net.set_qat_phase(fake_quant=True, observe=False, freeze_norm=True)
    assert all(m.fake_quant_enabled.item() == 1 for m in fqs(net))
    assert all(m.observer_enabled.item() == 0 for m in fqs(net))
    norms = [m for m in net.modules() if isinstance(m, DeploymentNorm)]
    assert not any(m.training for m in norms)
    net.set_qat_phase(fake_quant=False, observe=True, freeze_norm=False)
    assert all(m.training for m in norms)
    assert all(m.fake_quant_enabled.item() == 0 for m in fqs(net))


def test_deployed_files_load_into_fused_init_with_and_without_fused_flag():
    net = net_small()
    with torch.no_grad():  # non-trivial norm statistics
        for m in net.modules():
            if isinstance(m, DeploymentNorm):
                m.running_mean.uniform_(-0.5, 0.5)
                m.running_var.uniform_(0.5, 2.0)
    net.eval()
    x = torch.rand(1, 3, 10, 9)
    with torch.no_grad():
        ref = net(x)
        net.fuse_model()
        torch.testing.assert_close(net(x), ref, rtol=1e-4, atol=1e-5)
    sd = net.state_dict()
    old = {
        k: v for k, v in sd.items() if not k.endswith("fused_flag")
    }  # a pre-flag file
    for file in (sd, old):
        dep = net_small(fused_init=True).eval()
        dep.load_state_dict(file)  # strict
        with torch.no_grad():
            torch.testing.assert_close(dep(x), ref, rtol=1e-4, atol=1e-5)
    # a deployed file in an unfused-norm net still runs folded: the flag travels with the file
    flt = net_small()
    flt.fuse_reparam()
    flt.load_state_dict(sd)
    assert all(m.fused for m in flt.modules() if isinstance(m, DeploymentNorm))


# ------------------------------------------------------------------ config
def train_cfg(**train):
    return {
        "scale": S,
        "network_g": {"type": "aether_small"},
        "datasets": {"train": {"patch_size": P, "batch_size": B}},
        "train": {"optim_g": {"type": "adamw"}, "enable_qat": True, **train},
        "logger": {"total_iter": 100},
        "path": {},
    }


@pytest.mark.parametrize(
    ("train", "net", "match"),
    [
        ({"qat_type": "int4"}, {}, "qat_type"),
        ({"fused_init": False}, {}, "fused_init=false"),
        ({"quantize_residual": False}, {"quantize_residual": True}, "contradicts"),
        ({}, {"fused_init": True}, "deploy topology"),
        ({"calibration_iters": -1}, {}, "calibration_iters"),
        ({"calibration_iters": 70}, {}, "qat_freeze_norm_iter=70"),  # default 0.7 * 100
        ({"qat_freeze_observers_iter": 1}, {}, "qat_freeze_observers_iter=1"),
        ({"qat_freeze_norm_iter": 90, "qat_freeze_observers_iter": 50}, {}, "<="),
    ],
)
def test_validate_train_rejects_bad_qat_configs(train, net, match):
    cfg = train_cfg(**train)
    cfg["network_g"] |= net
    with pytest.raises(ConfigError, match=match):
        validate_train(cfg)


def test_validate_train_accepts_shipped_qat_keys():
    validate_train(train_cfg())
    validate_train(
        train_cfg(
            qat_type="int8",
            fused_init=True,
            quantize_residual=False,
            calibration_iters=5,
            qat_freeze_norm_iter=-1,
            qat_freeze_observers_iter=-1,
        )
    )
    cfg = train_cfg()
    cfg["network_g"]["fused_init"] = True
    cfg["path"]["pretrain_network_g"] = "x.pth"
    validate_train(cfg)


# ------------------------------------------------------------------ trainer
def qat_opt(tmp: Path, *, is_train=True, ema=0.9, **train):
    opt = {
        "name": "qat",
        "model_type": "image",
        "scale": S,
        "is_train": is_train,
        "network_g": {"type": "aether_small"},
        "datasets": {},
        "val": {
            "pbar": False,
            "metrics": {"psnr": {"type": "calculate_psnr", "crop_border": 0}},
        },
        "logger": {"total_iter": 8},
        "path": {
            "models": tmp / "models",
            "training_states": tmp / "training_states",
            "visualization": tmp / "vis",
        },
    }
    if is_train:
        opt["datasets"]["train"] = {
            "name": "t",
            "type": "paired",
            "patch_size": P,
            "batch_size": B,
        }
        opt["train"] = {
            "optim_g": {"type": "adamw", "lr": 1e-3},
            "pixel_opt": {"type": "L1Loss", "loss_weight": 1.0},
            "ema": ema,
            "enable_qat": True,
            "calibration_iters": 1,
            "qat_freeze_norm_iter": 3,
            "qat_freeze_observers_iter": 4,
        } | train
    return opt


def build(opt, seed=0):
    torch.manual_seed(seed)
    return build_model(opt, CPU)


def batches(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [
        {
            "lq": torch.rand(B, 3, P, P, generator=g),
            "gt": torch.rand(B, 3, P * S, P * S, generator=g),
        }
        for _ in range(n)
    ]


def run(model, data):
    for x in data:
        model.feed_data(x)
        model.optimize_parameters()


class ValSet(Dataset):
    def __init__(self) -> None:
        g = torch.Generator().manual_seed(7)
        self.opt = {"name": "val", "type": "paired"}
        self.items = [
            (torch.rand(3, 10, 9, generator=g), torch.rand(3, 20, 18, generator=g))
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        lq, gt = self.items[i]
        return {"lq": lq, "gt": gt, "lq_path": f"/x/img{i}.png"}


def test_enable_qat_needs_a_qat_capable_network(tmp_path):
    opt = qat_opt(tmp_path)
    opt["network_g"] = {"type": "compact", "num_feat": 8, "num_conv": 2}
    with pytest.raises(ConfigError, match="no QAT support"):
        build(opt)


def test_schedule_phases_follow_the_update_clock(tmp_path):
    m = build(qat_opt(tmp_path, ema=0))
    data = batches(5)
    seen, obs, stats = [], [], []
    for x in data:
        m.feed_data(x)
        m.optimize_parameters()
        seen.append(m._qat_phase)
        obs.append(observer_buffers(m.net_g))
        stats.append(norm_stats(m.net_g))
    # calibration 1, norms frozen from 3, observers from 4 (T5 D12/D13)
    assert seen == [
        (False, True, False),
        (True, True, False),
        (True, True, True),
        (True, False, True),
        (True, False, True),
    ]
    assert not same(obs[1], obs[2]) and same(obs[2], obs[3]) and same(obs[3], obs[4])
    assert (
        not same(stats[0], stats[1])
        and same(stats[1], stats[2])
        and same(stats[2], stats[4])
    )


def test_validation_freezes_observers_and_keeps_phase(tmp_path):
    m = build(qat_opt(tmp_path))
    run(m, batches(2))  # observers on in this phase
    flags = [x.observer_enabled.clone() for x in fqs(m.net_g)]
    before = observer_buffers(m.ema.module), observer_buffers(m.net_g)
    m.validation(DataLoader(ValSet(), batch_size=1), 2, None, save_img=False)
    assert same(before[0], observer_buffers(m.ema.module))
    assert same(before[1], observer_buffers(m.net_g))
    assert same(flags, [x.observer_enabled for x in fqs(m.net_g)])
    assert all(x.observer_enabled.item() == 1 for x in fqs(m.ema.module))  # restored


def test_a_skipped_first_update_leaves_a_valid_ema(tmp_path):
    m = build(qat_opt(tmp_path, calibration_iters=0))
    init = [p.clone() for p in m.ema.module.parameters()]
    step = m.g.step

    def nan_step():  # a non-finite gradient (fp16 overflow) after a finite, observed forward
        next(p for p in m.g.params if p.grad is not None).grad.fill_(float("nan"))
        return step()

    m.g.step = nan_step
    run(m, batches(1))
    assert m.g.skipped == 1
    m.check_finite()  # the EMA mirrors the observed ranges, not their +-inf reset
    assert same([*m.ema.module.buffers()], [*m.net_g.buffers()])
    assert same(init, [*m.ema.module.parameters()])
    m.validation(DataLoader(ValSet(), batch_size=1), 1, None, save_img=False)
    with observers_frozen(m.ema.module), torch.no_grad():
        assert m.ema.module(torch.rand(1, 3, 10, 9)).isfinite().all()
    m.g.step = step
    run(m, batches(1, seed=1))  # the first successful update still copies, never lerps
    assert same(tensors(m.net_g), tensors(m.ema.module))


def test_auto_resume_without_a_state_rejects_scratch_qat_on_folded_norms(tmp_path):
    opt = qat_opt(tmp_path, ema=0)
    opt["network_g"]["fused_init"] = True
    opt["auto_resume"] = True
    validate_train(opt)  # cannot know whether a state exists
    opt["path"]["resume_state"] = None  # what select_resume finds in an empty run
    with pytest.raises(ConfigError, match="deploy topology"):
        build(opt)


def test_save_resume_continues_exactly(tmp_path):
    data = batches(5)
    a = build(qat_opt(tmp_path / "a"))
    run(a, data)

    b = build(qat_opt(tmp_path / "b"))
    run(b, data[:2])
    b.save()
    opt = qat_opt(tmp_path / "b")
    opt["path"]["resume_state"] = tmp_path / "b" / "training_states" / "2.state"
    r = build(opt, seed=5)
    assert r.update == 2
    # restored as saved, before an update re-copies the EMA buffers (T5 A5)
    for x, y in ((b.net_g, r.net_g), (b.ema.module, r.ema.module)):
        assert same(tensors(x), tensors(y))
    x = torch.rand(1, 3, 10, 9)
    with (
        observers_frozen(b.ema.module),
        observers_frozen(r.ema.module),
        torch.no_grad(),
    ):
        assert torch.equal(b.ema.module(x), r.ema.module(x))
    r.restore_rng()
    run(r, data[2:])  # the phase is re-applied at the first update after the resume
    for x, y in ((a.net_g, r.net_g), (a.ema.module, r.ema.module)):
        assert same(tensors(x), tensors(y))
    sd = read_weights(tmp_path / "b" / "models" / "net_g_2.pth")
    assert aether.qat_state(sd) == {"quantize_residual": True}


def test_float_pretrain_then_qat_fine_tune(tmp_path):
    f = build(qat_opt(tmp_path / "f", enable_qat=False))
    run(f, batches(1))
    f.save()
    path = tmp_path / "f" / "models" / "net_g_1.pth"
    sd = read_weights(path)
    assert aether.qat_state(sd) is None
    opt = qat_opt(tmp_path / "q")
    opt["path"]["pretrain_network_g"] = str(path)
    q = build(opt)
    stats = [v for k, v in sd.items() if k.endswith(("running_mean", "running_var"))]
    assert same(stats, norm_stats(q.net_g))  # loaded before prepare, untouched by it
    run(q, batches(1))


def test_qat_export_as_pretrain_and_for_inference(tmp_path):
    t = build(qat_opt(tmp_path / "t"))
    run(t, batches(3))
    t.save()
    path = str(tmp_path / "t" / "models" / "net_g_3.pth")

    opt = qat_opt(tmp_path / "f")  # a QAT file loads after prepare_qat
    opt["path"]["pretrain_network_g"] = path
    m = build(opt)
    assert same(tensors(m.net_g), tensors(t.ema.module))
    opt["train"]["enable_qat"] = False
    with pytest.raises(ConfigError, match="QAT checkpoint"):
        build(opt)

    ref = t.ema.module
    ref.set_qat_phase(fake_quant=True, observe=False, freeze_norm=True)
    x = torch.rand(1, 3, 10, 9)
    for fused_init in (
        False,
        True,
    ):  # test.py: fake-quant eval, frozen ranges; any fused_init
        inf = qat_opt(tmp_path / "i", is_train=False)
        inf["network_g"]["fused_init"] = fused_init
        inf["path"]["pretrain_network_g"] = path
        m = build(inf)
        assert all(
            (q.fake_quant_enabled.item(), q.observer_enabled.item()) == (1, 0)
            for q in fqs(m.net_g)
        )
        with torch.no_grad():
            assert torch.equal(m.net_g.eval()(x), ref(x))


def test_log_macs_leaves_qat_observers_and_ranges_alone(monkeypatch):
    """The MAC probe runs on noise before restore_rng(): it must not move activation ranges."""
    import sys  # noqa: PLC0415
    import types  # noqa: PLC0415

    import train  # noqa: PLC0415

    def profile_macs(net, x):  # torchprofile traces: a real forward
        net(x)
        return 1

    monkeypatch.setitem(
        sys.modules, "torchprofile", types.SimpleNamespace(profile_macs=profile_macs)
    )
    net = net_small()
    net.prepare_qat()
    observed(net)  # observers on, ranges set
    sd = {k: v.clone() for k, v in net.state_dict().items()}
    model = argparse.Namespace(net_g=net, net_d=None, device=torch.device("cpu"))
    train.log_macs(model, {"input_size": 8})
    assert all(m.observer_enabled.item() == 1 for m in fqs(net))
    assert all(torch.equal(v, sd[k]) for k, v in net.state_dict().items())


# ------------------------------------------------------------------ convert.py
@pytest.fixture
def convert_py():
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import convert  # noqa: PLC0415

    return convert


def test_convert_exports_a_verified_qdq_onnx(tmp_path, convert_py):
    import onnx  # noqa: PLC0415

    net = net_small()
    net.prepare_qat()
    net.train()
    observed(net)
    torch.save({"params": net.state_dict()}, tmp_path / "q.pth")
    argv = ["--input", str(tmp_path / "q.pth"), "-net", "aether_small", "-s", str(S)]
    assert convert_py.main(argv) == 0  # parity is checked inside (verify_qat)
    mdl = onnx.load(tmp_path / "q_fp32.onnx")
    ops = Counter(n.op_type for n in mdl.graph.node)
    assert ops["QuantizeLinear"] == ops["DequantizeLinear"] == 226
    prod = {o: n.op_type for n in mdl.graph.node for o in n.output}
    convs = [n for n in mdl.graph.node if n.op_type == "Conv"]
    qdq = [
        n for n in convs if all(prod.get(i) == "DequantizeLinear" for i in n.input[:2])
    ]
    assert (len(qdq), len(convs)) == (71, 74)
    assert "Pad" not in ops
    inits = {i.name: i for i in mdl.graph.initializer}
    alias = {n.output[0]: n.input[0] for n in mdl.graph.node if n.op_type == "Identity"}
    qparams = [
        alias.get(i, i)
        for n in mdl.graph.node
        if n.op_type.endswith("QuantizeLinear")
        for i in n.input[1:3]
    ]
    assert all(i in inits for i in qparams)  # constants, folded
    assert not [
        i for i in qparams if list(inits[i].dims) == [1]
    ]  # per-tensor ones are 0-d


@pytest.mark.parametrize(
    ("ref", "out"),
    [(0.0, float("nan")), (0.0, float("inf")), (float("nan"), float("nan"))],
)
def test_qdq_verification_fails_on_non_finite_outputs(
    convert_py, monkeypatch, ref, out
):
    import onnxruntime  # noqa: PLC0415

    class Session:
        def __init__(self, *_a, **_k) -> None:
            pass

        def get_inputs(self):
            return [argparse.Namespace(name="input")]

        def run(self, _names, feed):
            return [np.full_like(feed["input"], out)]

    monkeypatch.setattr(onnxruntime, "InferenceSession", Session)
    args = argparse.Namespace(static=None)
    with pytest.raises(RuntimeError, match="deviates"):
        convert_py.verify_qat(args, "unused.onnx", lambda x: torch.full_like(x, ref))


def test_convert_float_aether_exports_unquantized(tmp_path, convert_py):
    torch.save({"params": net_small().state_dict()}, tmp_path / "f.pth")
    argv = ["--input", str(tmp_path / "f.pth"), "-net", "aether_small", "-s", str(S)]
    assert convert_py.main(argv) == 0


def qat_file(tmp_path) -> tuple[aether, Path]:
    net = net_small()
    net.prepare_qat()
    net.train()
    observed(net)
    torch.save({"params": net.state_dict()}, tmp_path / "q.pth")
    return net, tmp_path / "q.pth"


def strip_argv(path: Path) -> list[str]:
    return ["--input", str(path), "-net", "aether_small", "-s", str(S), "--strip-qat"]


def onnx_ops(path: Path) -> Counter:
    import onnx  # noqa: PLC0415

    return Counter(n.op_type for n in onnx.load(path).graph.node)


def test_strip_qat_saves_a_float_checkpoint_for_fused_init(tmp_path, convert_py):
    qat, path = qat_file(tmp_path)
    assert convert_py.main([*strip_argv(path), "--onnx"]) == 0  # parity checked inside
    sd = read_weights(str(tmp_path / "q_float.pth"))
    assert not [k for k in sd if "fake_quant" in k or "activation_post_process" in k]
    ops = onnx_ops(tmp_path / "q_float_fp32.onnx")
    assert ops["Conv"]
    assert not ops["QuantizeLinear"]
    assert not ops["DequantizeLinear"]

    qat.eval()
    qat.set_qat_phase(fake_quant=True, observe=False, freeze_norm=True)
    qat.fold_norms()
    x = torch.rand(1, 3, 10, 9)
    with torch.no_grad():
        quant = qat(x)
    qat.set_qat_phase(fake_quant=False, observe=False, freeze_norm=True)
    inf = qat_opt(
        tmp_path / "i", is_train=False
    )  # test.py: strict load into the deploy topology
    inf["network_g"]["fused_init"] = True
    inf["path"]["pretrain_network_g"] = str(tmp_path / "q_float.pth")
    m = build(inf)
    assert not any(isinstance(x, tq.FakeQuantizeBase) for x in m.net_g.modules())
    with torch.no_grad():
        out = m.net_g.eval()(x)
        torch.testing.assert_close(out, qat(x), rtol=1e-4, atol=1e-5)
    assert (out - quant).abs().max() <= 1e-2  # A9


def test_stripped_checkpoint_reopens_in_a_separate_convert_call(tmp_path, convert_py):
    _, path = qat_file(tmp_path)
    assert convert_py.main(strip_argv(path)) == 0
    flt = tmp_path / "q_float.pth"
    assert convert_py.main(strip_argv(flt)) == 2  # not QAT: rejected before any load
    assert not (tmp_path / "q_float_float.pth").exists()
    assert (
        convert_py.main(strip_argv(flt)[:-1]) == 0
    )  # plain export, fused topology detected
    ops = onnx_ops(tmp_path / "q_float_fp32.onnx")
    assert ops["Conv"]
    assert not ops["QuantizeLinear"]


def test_strip_qat_rejects_a_float_checkpoint(tmp_path, convert_py):
    torch.save({"params": net_small().state_dict()}, tmp_path / "f.pth")
    argv = ["--input", str(tmp_path / "f.pth"), "-net", "aether_small", "-s", str(S)]
    assert convert_py.main([*argv, "--strip-qat"]) == 2
    assert not (tmp_path / "f_float.pth").exists()


def test_strip_qat_rejects_a_float_net_far_from_fake_quant(tmp_path, convert_py):
    net = net_small()
    net.prepare_qat()
    net.train()
    observed(net)
    with (
        torch.no_grad()
    ):  # weights beyond the observed int8 range: fake quant clips them
        net.conv_last.weight.mul_(10)
    torch.save({"params": net.state_dict()}, tmp_path / "q.pth")
    assert convert_py.main(strip_argv(tmp_path / "q.pth")) == 1
    assert not (tmp_path / "q_float.pth").exists()
    assert convert_py.main([*strip_argv(tmp_path / "q.pth"), "--nocheck"]) == 0
