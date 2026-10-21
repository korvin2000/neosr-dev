"""Entry points (design T2 §9 D1, T1 §6 B3, T3 §5 Phase 2 error behaviour): main() exit codes, the
training loop, resume selection, graceful stop, test.py and convert.py. CPU, tiny nets, synthetic PNGs."""

import importlib.util
import logging
import os
import signal
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from neosr.config import ConfigError
from neosr.data.prefetch_dataloader import CUDAPrefetcher
from neosr.models.image import image
from neosr.runtime import Runtime

ROOT = Path(__file__).resolve().parents[1]
CPU = Runtime(torch.device("cpu"), 0, 1, 0, False)
NET_G_2000 = Path(
    os.environ.get(
        "NEOSR_NET_G_2000",
        "/home/korvin/apps/neosr-dev/experiments/train_esrgan/models/net_g_2000.pth",
    )
)


def _load(name: str) -> types.ModuleType:
    """train.py / test.py / convert.py as modules (`test` would shadow the stdlib package)."""
    spec = importlib.util.spec_from_file_location(
        f"neosr_entry_{name}", ROOT / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


train_py, test_py, convert_py = _load("train"), _load("test"), _load("convert")


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def data(tmp_path_factory) -> Path:
    """6 train pairs (lq 16 px, gt 32 px, scale 2) and 2 val pairs."""
    root = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(0)
    for split, n in (("train", 6), ("val", 2)):
        for kind in ("lq", "gt"):
            (root / split / kind).mkdir(parents=True)
        for i in range(n):
            gt = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
            cv2.imwrite(str(root / split / "gt" / f"im{i}.png"), gt)
            lq = cv2.resize(gt, (16, 16), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(root / split / "lq" / f"im{i}.png"), lq)
    return root


@pytest.fixture(autouse=True)
def _restore_globals():
    """setup_runtime flips process-wide flags; the loop installs a SIGINT handler; the logger keeps a file."""
    env = dict(os.environ)
    det = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    cudnn = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    fill = torch.utils.deterministic.fill_uninitialized_memory  # type: ignore[attr-defined]
    handler = signal.getsignal(signal.SIGINT)
    yield
    os.environ.clear()
    os.environ.update(env)
    torch.use_deterministic_algorithms(det[0], warn_only=det[1])
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn
    torch.utils.deterministic.fill_uninitialized_memory = fill  # type: ignore[attr-defined]
    signal.signal(signal.SIGINT, handler)
    log = logging.getLogger("neosr")
    for h in [h for h in log.handlers if isinstance(h, logging.FileHandler)]:
        log.removeHandler(h)
        h.close()


@pytest.fixture
def cpu(monkeypatch):
    """Run main() on the CPU: setup_runtime picks the device from torch.cuda.is_available()."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def toml(
    tmp: Path,
    data: Path,
    *,
    total=4,
    save=2,
    val_freq=2,
    batch=2,
    accumulate=1,
    extra="",
    path_extra="",
) -> Path:
    text = f"""
name = "tiny"
model_type = "image"
scale = 2
manual_seed = 0

[datasets.train]
type = "paired"
dataroot_gt = "{data / "train" / "gt"}"
dataroot_lq = "{data / "train" / "lq"}"
patch_size = 8
batch_size = {batch}
accumulate = {accumulate}
num_worker_per_gpu = 0

[datasets.val]
name = "valset"
type = "paired"
dataroot_gt = "{data / "val" / "gt"}"
dataroot_lq = "{data / "val" / "lq"}"

[val]
val_freq = {val_freq}
pbar = false
[val.metrics.psnr]
type = "calculate_psnr"
crop_border = 0

[path]
experiments_root = "{tmp / "exp"}"
{path_extra}

[network_g]
type = "compact"
num_feat = 8
num_conv = 2

[train]
[train.optim_g]
type = "adamw"
lr = 1e-3
[train.pixel_opt]
type = "L1Loss"
loss_weight = 1.0

[logger]
total_iter = {total}
print_freq = 1
save_checkpoint_freq = {save}
{extra}
"""
    f = tmp / f"cfg_{abs(hash(text))}.toml"
    f.write_text(text)
    return f


def run_dir(tmp: Path) -> Path:
    return tmp / "exp" / "tiny"


def states(tmp: Path) -> list[int]:
    return sorted(
        int(p.stem) for p in (run_dir(tmp) / "training_states").glob("*.state")
    )


def logs(tmp: Path) -> str:
    return "".join(p.read_text() for p in run_dir(tmp).glob("train_*.log"))


def archived(tmp: Path) -> list[Path]:
    return list((tmp / "exp").glob("tiny_archived_*"))


# ------------------------------------------------------------------ train.py: loop, save, resume
def test_train_save_validate_resume(tmp_path, data, cpu):
    cfg = toml(tmp_path, data)
    assert train_py.main(["-opt", str(cfg)]) == 0
    d = run_dir(tmp_path)
    assert states(tmp_path) == [2, 4]
    models = {p.name for p in (d / "models").iterdir()}
    assert {"net_g_2.pth", "net_g_4.pth", "net_g_latest.pth"} <= models
    assert not list(d.rglob("*.tmp"))
    # periodic validation at 2 and 4, plus the final one after a natural end (also at 4)
    assert (d / "visualization" / "im0" / "im0_2.png").is_file()
    assert (d / "visualization" / "im0" / "im0_4.png").is_file()
    assert list(d.glob("cfg_*.toml"))  # config copy with provenance header
    assert list(d.glob("cfg_*.toml"))[0].read_text().startswith("# GENERATE TIME")
    text = logs(tmp_path)
    assert "[ iter:      4 ]" in text and "Best:" in text

    state = torch.load(d / "training_states" / "4.state", weights_only=True)
    assert state["update"] == 4
    assert state["best"]["valset"]["psnr"]["iter"] in {2, 4}

    # auto-resume to 6: continues from 4.state, never archives the run folder
    cfg = toml(tmp_path, data, total=6)
    assert train_py.main(["-opt", str(cfg), "--auto_resume"]) == 0
    assert states(tmp_path) == [2, 4, 6]
    assert archived(tmp_path) == []
    assert (
        torch.load(d / "training_states" / "6.state", weights_only=True)["update"] == 6
    )

    # a fresh run (no resume) archives the existing folder (status quo)
    assert train_py.main(["-opt", str(toml(tmp_path, data, total=2))]) == 0
    assert len(archived(tmp_path)) == 1
    assert states(tmp_path) == [2]


def test_resume_at_total_returns_before_iterating(tmp_path, data, cpu, monkeypatch):
    assert train_py.main(["-opt", str(toml(tmp_path, data, total=2))]) == 0

    def boom(self):
        raise AssertionError("prefetcher.reset() called")

    monkeypatch.setattr(CUDAPrefetcher, "reset", boom)
    assert (
        train_py.main(["-opt", str(toml(tmp_path, data, total=2)), "--auto_resume"])
        == 0
    )
    assert "nothing to do: update 2 >= total_iter 2" in logs(tmp_path)
    assert states(tmp_path) == [2]


def test_explicit_missing_resume_raises_without_archiving(tmp_path, data, cpu):
    d = run_dir(tmp_path)
    d.mkdir(parents=True)
    (d / "keep.txt").write_text("x")
    missing = tmp_path / "nope.state"
    cfg = toml(tmp_path, data, path_extra=f'resume_state = "{missing}"')
    assert train_py.main(["-opt", str(cfg)]) == 1
    assert (d / "keep.txt").is_file()
    assert archived(tmp_path) == []


def test_explicit_resume_into_a_new_run_root(tmp_path, data, cpu):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    assert train_py.main(["-opt", str(toml(a, data, total=2))]) == 0
    copied = tmp_path / "copied.state"
    copied.write_bytes((run_dir(a) / "training_states" / "2.state").read_bytes())
    cfg = toml(b, data, total=4, path_extra=f'resume_state = "{copied}"')
    assert not run_dir(b).exists()
    assert train_py.main(["-opt", str(cfg)]) == 0
    assert states(b) == [4]
    assert archived(b) == []
    assert "resumed copied.state: update 2" in logs(b)


def test_corrupt_newest_state_raises_naming_it(tmp_path, data, cpu):
    assert train_py.main(["-opt", str(toml(tmp_path, data, total=2))]) == 0
    bad = run_dir(tmp_path) / "training_states" / "10.state"
    bad.write_bytes(b"not a checkpoint")
    assert train_py.main(["-opt", str(toml(tmp_path, data)), "--auto_resume"]) == 1
    assert str(bad) in logs(tmp_path)
    assert archived(tmp_path) == []
    assert states(tmp_path) == [2, 10]


def test_zero_batch_train_loader_is_config_error(tmp_path, data, cpu):
    assert train_py.main(["-opt", str(toml(tmp_path, data, batch=8))]) == 2


def test_build_loaders_rejects_zero_batches(tmp_path, data):
    from neosr.config import load_config  # noqa: PLC0415

    cfg = load_config(toml(tmp_path, data, batch=8), is_train=True, root=ROOT)
    with pytest.raises(ConfigError, match="yields no batch"):
        train_py.build_loaders(cfg, CPU)


# ------------------------------------------------------------------ T3 §5 phase 2: error behaviour
def test_malformed_toml_returns_2(tmp_path, capsys):
    f = tmp_path / "bad.toml"
    f.write_text("name = [unclosed\n")
    assert train_py.main(["-opt", str(f)]) == 2
    assert "config error" in capsys.readouterr().err


def test_missing_pretrain_returns_1_with_path_in_log(tmp_path, data, cpu):
    missing = tmp_path / "missing_g.pth"
    cfg = toml(tmp_path, data, path_extra=f'pretrain_network_g = "{missing}"')
    assert train_py.main(["-opt", str(cfg)]) == 1
    text = logs(tmp_path)
    assert "FileNotFoundError" in text and str(missing) in text
    assert text.count("training failed") == 1  # traceback logged once


def test_constructor_body_type_error_returns_1(tmp_path, data, cpu, monkeypatch):
    from neosr.archs import compact_arch  # noqa: PLC0415

    def broken(self, **kwargs):
        raise TypeError("bug inside the constructor")

    monkeypatch.setattr(compact_arch.compact, "__init__", broken)
    assert train_py.main(["-opt", str(toml(tmp_path, data))]) == 1
    assert "bug inside the constructor" in logs(tmp_path)


# ------------------------------------------------------------------ graceful stop (T2 §5.2, T1 B3)
def _sigint_after(monkeypatch, micro_batches: set[int]):
    """Deliver SIGINT to this process after the given micro-batches (1-based) of optimize_parameters."""
    real, seen = image.optimize_parameters, [0]

    def spy(self):
        out = real(self)
        seen[0] += 1
        if seen[0] in micro_batches:
            os.kill(os.getpid(), signal.SIGINT)
        return out

    monkeypatch.setattr(image, "optimize_parameters", spy)
    return seen


def test_ctrl_c_at_boundary_saves_once_and_exits_0(tmp_path, data, cpu, monkeypatch):
    _sigint_after(monkeypatch, {3})
    cfg = toml(tmp_path, data, total=100, save=50, val_freq=1)
    assert train_py.main(["-opt", str(cfg)]) == 0
    d = run_dir(tmp_path)
    assert states(tmp_path) == [3]
    assert (d / "models" / "net_g_latest.pth").is_file()
    assert "stop requested: saving update 3" in logs(tmp_path)
    vis = d / "visualization" / "im0"
    assert {p.name for p in vis.glob("im0_[0-9]*.png")} == {
        "im0_1.png",
        "im0_2.png",
    }  # none after the stop


def test_ctrl_c_mid_window_saves_at_the_next_boundary(tmp_path, data, cpu, monkeypatch):
    _sigint_after(monkeypatch, {3})  # micro-batch 3 opens window 2 (accumulate 2)
    cfg = toml(tmp_path, data, total=100, save=50, val_freq=50, accumulate=2)
    assert train_py.main(["-opt", str(cfg)]) == 0
    assert states(tmp_path) == [2]  # nothing saved mid-window
    state = torch.load(
        run_dir(tmp_path) / "training_states" / "2.state", weights_only=True
    )
    assert state["update"] == 2


def test_second_ctrl_c_returns_130_and_keeps_last_checkpoint(
    tmp_path, data, cpu, monkeypatch
):
    _sigint_after(
        monkeypatch, {3}
    )  # first: stop requested at update 2 (accumulate 2) ...
    real_save = image.save

    def save_interrupted(self, *, final=False):
        os.kill(
            os.getpid(), signal.SIGINT
        )  # ... second, before the final save writes anything
        real_save(self, final=final)

    monkeypatch.setattr(image, "save", save_interrupted)
    cfg = toml(tmp_path, data, total=100, save=50, val_freq=50, accumulate=2)
    assert train_py.main(["-opt", str(cfg)]) == 130
    assert states(tmp_path) == []
    assert not list(run_dir(tmp_path).rglob("*.tmp"))


@pytest.mark.parametrize("accumulate", [1, 2])
def test_consumed_samples_unique_per_epoch(
    tmp_path, data, cpu, monkeypatch, accumulate
):
    """6 images, batch 2: 3 batches per epoch, so accumulate 2 leaves a window across each epoch end."""
    seen: list[tuple[int, list[str]]] = []
    real = image.feed_data

    def spy(self, batch):
        seen.append((self.epoch, list(batch["gt_path"])))
        real(self, batch)

    monkeypatch.setattr(image, "feed_data", spy)
    total = 5
    cfg = toml(tmp_path, data, total=total, save=50, val_freq=50, accumulate=accumulate)
    assert train_py.main(["-opt", str(cfg)]) == 0
    assert len(seen) == total * accumulate  # no batch re-fed, none skipped
    for epoch in {e for e, _ in seen}:
        ids = [i for e, paths in seen if e == epoch for i in paths]
        assert len(ids) == len(set(ids))
    assert len({e for e, _ in seen}) >= 2  # crossed an epoch boundary


# ------------------------------------------------------------------ MAC profiling (T2 D28)
def test_log_macs_uses_bare_nets_in_eval_and_restores_flags(
    tmp_path, data, monkeypatch
):
    from neosr.config import load_config  # noqa: PLC0415
    from neosr.models import build_model  # noqa: PLC0415

    calls = []

    def profile_macs(net, x):
        calls.append((
            type(net).__name__,
            net.training,
            torch.is_grad_enabled(),
            tuple(x.shape),
        ))
        return 123

    monkeypatch.setitem(
        sys.modules, "torchprofile", types.SimpleNamespace(profile_macs=profile_macs)
    )
    cfg = load_config(toml(tmp_path, data), is_train=True, root=ROOT)
    cfg["input_size"] = 8
    model = build_model(cfg, CPU)
    rng = torch.get_rng_state()
    train_py.log_macs(model, cfg)
    assert calls == [("compact", False, False, (1, 3, 8, 8))]
    assert model.net_g.training
    assert torch.equal(torch.get_rng_state(), rng)
    cfg.pop("input_size")
    train_py.log_macs(model, cfg)  # no input_size: nothing profiled
    assert len(calls) == 1


# ------------------------------------------------------------------ DDP: check_finite on all ranks
def _ddp_worker(rank: int, cfg_path: str, init: str, out: str) -> None:
    import torch.distributed as dist  # noqa: PLC0415

    from neosr.config import load_config  # noqa: PLC0415
    from neosr.runtime import seed_all  # noqa: PLC0415

    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=2)
    calls = []
    real = image.check_finite

    def spy(self):
        calls.append(self.update)
        real(self)

    image.check_finite = spy
    trainer = _load("train")
    cfg = load_config(cfg_path, is_train=True, root=ROOT)
    cfg["path"]["resume_state"] = None
    seed_all(rank)
    trainer.train(cfg, Runtime(torch.device("cpu"), rank, 2, 0, True))
    torch.save(calls, f"{out}.{rank}")
    dist.destroy_process_group()


def test_ddp_gloo_check_finite_on_all_ranks(tmp_path, data):
    cfg = toml(tmp_path, data, total=4, save=2, val_freq=2)
    (run_dir(tmp_path) / "training_states").mkdir(parents=True)
    (run_dir(tmp_path) / "models").mkdir(parents=True)
    out = str(tmp_path / "calls")
    torch.multiprocessing.spawn(
        _ddp_worker,
        args=(str(cfg), f"file://{tmp_path / 'store'}", out),
        nprocs=2,
        join=True,
    )
    for r in (0, 1):
        assert torch.load(f"{out}.{r}") == [
            2,
            4,
        ]  # periodic at 2, done branch at 4: no hang
    assert states(tmp_path) == [2, 4]


# ------------------------------------------------------------------ test.py
def test_test_py_main_writes_outputs(tmp_path, data, cpu):
    pth = tmp_path / "g.pth"
    from neosr.archs import build_network  # noqa: PLC0415

    net = build_network({"type": "compact", "num_feat": 8, "num_conv": 2}, scale=2)
    torch.save({"params": net.state_dict()}, pth)
    f = tmp_path / "infer.toml"
    f.write_text(f"""
name = "tinytest"
model_type = "image"
scale = 2
[datasets.test_1]
name = "valset"
type = "paired"
dataroot_gt = "{data / "val" / "gt"}"
dataroot_lq = "{data / "val" / "lq"}"
[val]
pbar = false
[val.metrics.psnr]
type = "calculate_psnr"
crop_border = 0
[path]
pretrain_network_g = "{pth}"
results_root = "{tmp_path / "results"}"
[network_g]
type = "compact"
num_feat = 8
num_conv = 2
""")
    assert test_py.main(["-opt", str(f)]) == 0
    out = tmp_path / "results" / "tinytest" / "valset"
    assert sorted(p.name for p in out.iterdir()) == [
        "im0_tinytest.png",
        "im1_tinytest.png",
    ]


def test_test_py_empty_dataset_returns_2(tmp_path, cpu, capsys):
    for kind in ("lq", "gt"):
        (tmp_path / kind).mkdir()
    pth = tmp_path / "g.pth"
    from neosr.archs import build_network  # noqa: PLC0415

    net = build_network({"type": "compact", "num_feat": 8, "num_conv": 2}, scale=2)
    torch.save({"params": net.state_dict()}, pth)
    f = tmp_path / "infer.toml"
    f.write_text(f"""
name = "empty"
model_type = "image"
scale = 2
[datasets.test_1]
name = "none"
type = "paired"
dataroot_gt = "{tmp_path / "gt"}"
dataroot_lq = "{tmp_path / "lq"}"
[val]
pbar = false
[val.metrics.psnr]
type = "calculate_psnr"
crop_border = 0
[path]
pretrain_network_g = "{pth}"
results_root = "{tmp_path / "results"}"
[network_g]
type = "compact"
num_feat = 8
num_conv = 2
""")
    assert test_py.main(["-opt", str(f)]) == 2
    assert "[datasets.test_1] has no images" in capsys.readouterr().err


@pytest.mark.skipif(
    not NET_G_2000.is_file(),
    reason="needs experiments/train_esrgan/models/net_g_2000.pth",
)
def test_test_py_reproduces_old_loader_psnr(tmp_path, data):
    """net_g_2000.pth through read_weights (test.py) gives the PSNR of the old loader's weights exactly."""
    from neosr.archs import build_network  # noqa: PLC0415
    from neosr.config import load_config  # noqa: PLC0415
    from neosr.metrics.calculate import calculate_psnr  # noqa: PLC0415
    from neosr.utils import tensor2img  # noqa: PLC0415

    # scale-4 val pairs
    for kind in ("lq", "gt"):
        (tmp_path / kind).mkdir()
    rng = np.random.default_rng(1)
    for i in range(2):
        gt = rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "gt" / f"x{i}.png"), gt)
        cv2.imwrite(
            str(tmp_path / "lq" / f"x{i}.png"),
            cv2.resize(gt, (12, 12), interpolation=cv2.INTER_AREA),
        )
    f = tmp_path / "infer.toml"
    f.write_text(f"""
name = "esr"
model_type = "image"
scale = 4
[datasets.test_1]
name = "set"
type = "paired"
dataroot_gt = "{tmp_path / "gt"}"
dataroot_lq = "{tmp_path / "lq"}"
[val]
pbar = false
save_img = false
[val.metrics.psnr]
type = "calculate_psnr"
crop_border = 4
[path]
pretrain_network_g = "{NET_G_2000}"
results_root = "{tmp_path / "results"}"
[network_g]
type = "esrgan"
""")
    model = test_py.test(load_config(f, is_train=False, root=ROOT), CPU)
    new = model.best_metric_results["set"]["psnr"]["val"]

    # the old loader (base.load_network @55f0cdd): key precedence, strip "module.", strict load
    obj = torch.load(NET_G_2000, map_location="cpu", weights_only=True)
    key = next(k for k in ("params-ema", "params_ema", "params") if k in obj)
    sd = {k.removeprefix("module."): v for k, v in obj[key].items()}
    net = build_network({"type": "esrgan"}, scale=4)
    net.load_state_dict(sd, strict=True)
    net.eval()
    vals = []
    for i in range(2):
        lq = (
            cv2.imread(str(tmp_path / "lq" / f"x{i}.png"))[..., ::-1].astype(np.float32)
            / 255.0
        )
        lq = torch.from_numpy(lq.copy()).permute(2, 0, 1)[None].float()
        gt = cv2.imread(str(tmp_path / "gt" / f"x{i}.png"))
        with torch.inference_mode():
            vals.append(calculate_psnr(tensor2img([net(lq)]), gt, crop_border=4))
    assert new == sum(vals) / len(vals)


# ------------------------------------------------------------------ convert.py (T3 D26)
def _convert_args(inp: Path, out: Path, *extra: str):
    from neosr.cli import convert_args  # noqa: PLC0415

    return convert_args([
        "--input",
        str(inp),
        "-net",
        "compact",
        "-s",
        "2",
        "--output",
        str(out),
        *extra,
    ])


@pytest.mark.parametrize("layout", ["params", "orig_mod"])
def test_convert_load_net_reads_exports_and_prefixed_dicts(tmp_path, layout):
    from neosr.archs import build_network  # noqa: PLC0415

    torch.manual_seed(0)
    net = build_network({"type": "compact"}, scale=2)
    sd = net.state_dict()
    obj = (
        {"params": sd}
        if layout == "params"
        else {f"_orig_mod.module.{k}": v for k, v in sd.items()}
    )
    torch.save(obj, tmp_path / "w.pth")
    loaded = convert_py.load_net(_convert_args(tmp_path / "w.pth", tmp_path / "m.onnx"))
    assert not loaded.training
    for k, v in loaded.state_dict().items():
        assert v.device.type == "cpu"
        assert torch.equal(v, sd[k])


def test_convert_load_net_plainusr_loads_unfused(tmp_path):
    from neosr.archs import build_network  # noqa: PLC0415
    from neosr.cli import convert_args  # noqa: PLC0415

    sd = build_network({"type": "plainusr"}, scale=2).state_dict()
    torch.save({"params": sd}, tmp_path / "w.pth")
    args = convert_args([
        "--input",
        str(tmp_path / "w.pth"),
        "-net",
        "plainusr",
        "-s",
        "2",
        "--output",
        str(tmp_path / "m.onnx"),
        "--prune",
    ])
    net = convert_py.load_net(args)
    assert not net.training
    assert net.state_dict().keys() == sd.keys()
    with torch.inference_mode():
        assert net(torch.rand(1, 3, 8, 8)).shape == (1, 3, 16, 16)


@pytest.mark.parametrize("out", ["m.onnx", None])
def test_convert_main_exports_verified_onnx_on_cpu(tmp_path, out):
    from neosr.archs import build_network  # noqa: PLC0415

    torch.save(
        {"params": build_network({"type": "compact"}, scale=2).state_dict()},
        tmp_path / "w.pth",
    )
    argv = ["--input", str(tmp_path / "w.pth"), "-net", "compact", "-s", "2"]
    if out is not None:
        argv += ["--output", str(tmp_path / out)]
    assert convert_py.main(argv) == 0
    # without --output the export lands next to the input
    assert (tmp_path / ("m_fp32.onnx" if out else "w_fp32.onnx")).is_file()


def test_convert_unknown_network_returns_2(tmp_path):
    torch.save({"params": {}}, tmp_path / "w.pth")
    assert (
        convert_py.main([
            "--input",
            str(tmp_path / "w.pth"),
            "-net",
            "nonexistent_arch",
        ])
        == 2
    )
