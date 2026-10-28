"""Stage 2, metrics lane: exact on-device validation metrics (MT-04), tensor2img (MT-08), metric-net
lifecycle (MT-05, MT-13, MT-12), validation runtime (MT-11, T-03, MT-10, MT-15) and the ONNX converter
(MT-02, MT-07, MT-16, MT-17), plus the review fixes of stage 2."""

import logging
import math
import threading
import weakref
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid

from neosr.metrics import build_metric, calculate
from neosr.models import build_model
from neosr.models import image as image_mod
from neosr.runtime import Runtime
from neosr.utils.img_util import quantize, tensor2img

CPU = Runtime(torch.device("cpu"), 0, 1, 0, False)
CUDA = torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])


def old_tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """tensor2img before MT-08 (f3d2b7c), the reference for the in-place rewrite."""
    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for tensor_ in tensor:
        t = tensor_.squeeze(0) if tensor_.dim() == 4 else tensor_
        n_tensor = t.detach().float().cpu().clamp(*min_max)
        n_tensor = (n_tensor - min_max[0]) / (min_max[1] - min_max[0])
        n_dim = n_tensor.dim()
        if n_dim == 4:
            img_np = make_grid(
                n_tensor, nrow=int(math.sqrt(n_tensor.size(0))), normalize=False
            ).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = n_tensor.numpy().transpose(1, 2, 0)
            if img_np.shape[2] == 1:
                img_np = np.squeeze(img_np, axis=2)
            elif rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_np = n_tensor.numpy()
        if out_type == np.uint8:
            img_np = (img_np * 255.0).round()
        result.append(img_np.astype(out_type))
    return result[0] if len(result) == 1 else result


def _noisy(shape, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(shape, generator=g) * 0.4 + 0.5  # ~11 % outside [0, 1]
    t.view(-1)[:4] = torch.tensor([
        0.5 / 255,
        1.5 / 255,
        2.5 / 255,
        254.5 / 255,
    ])  # ties
    return t.to(dtype)


# ---------------------------------------------------------------------------- MT-08 tensor2img
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 3, 17, 23),
        (3, 9, 5),
        (1, 1, 6, 7),
        (1, 4, 6, 5),
        (1, 6),
        (5, 6),
        (4, 3, 6, 5),
    ],
)
@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"rgb2bgr": False},
        {"min_max": (-1, 1)},
        {"out_type": np.float32},
        {"out_type": np.float32, "rgb2bgr": False, "min_max": (-1, 1)},
    ],
)
def test_tensor2img_is_identical_to_the_old_implementation(dtype, shape, kw):
    t = _noisy(shape, dtype)
    before = t.clone()
    new, old = tensor2img(t, **kw), old_tensor2img(t, **kw)
    assert new.dtype == old.dtype
    assert new.shape == old.shape
    assert new.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(new, old)
    assert torch.equal(t, before)  # the caller's tensor is untouched (fp32 CPU aliases)


def test_tensor2img_list_input():
    ts = [_noisy((1, 3, 5, 4), seed=s) for s in range(2)]
    for a, b in zip(tensor2img(ts), old_tensor2img(ts), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_quantize_is_tensor2img_on_any_device(device, dtype):
    t = _noisy((1, 3, 31, 29), dtype).to(device)
    before = t.clone()
    q = quantize(t)
    assert q.dtype == torch.float32
    assert q.device == t.device
    assert torch.equal(t, before)
    host = q[0].to(torch.uint8).permute(1, 2, 0).cpu().numpy()[..., ::-1]
    np.testing.assert_array_equal(host, old_tensor2img(t))


# ---------------------------------------------------------------------------- MT-04 tensor metrics
def _pair(c, h, w, seed):
    g = torch.Generator().manual_seed(seed)
    gt = torch.rand(1, c, h, w, generator=g)
    sr = gt + torch.randn(1, c, h, w, generator=g) * 0.1
    return sr, gt


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("c", [3, 1])
@pytest.mark.parametrize("crop_border", [0, 4])
@pytest.mark.parametrize("y", [False, True])
@pytest.mark.parametrize("metric", ["psnr", "ssim"])
def test_tensor_metrics_reproduce_the_numpy_reference(
    device, c, crop_border, y, metric
):
    """quantise on the device, then fp64 statistics: the numpy reference to <= 1e-10."""
    for seed, (h, w) in enumerate([(48, 40), (31, 57), (96, 96)]):
        sr, gt = _pair(c, h, w, seed)
        ref = getattr(calculate, f"calculate_{metric}")(
            tensor2img(sr), tensor2img(gt), crop_border=crop_border, test_y_channel=y
        )
        f = build_metric(
            {
                "type": f"calculate_{metric}",
                "crop_border": crop_border,
                "test_y_channel": y,
            },
            torch.device(device),
        )
        val = f(quantize(sr.to(device)), quantize(gt.to(device)))
        assert val.device.type == device
        assert val.dtype == torch.float64
        assert abs(val.item() - ref) <= 1e-10


@pytest.mark.parametrize("metric", ["psnr", "ssim"])
def test_tensor_metrics_accept_non_contiguous_input(metric):
    """img2tensor and channels_last outputs are HWC-strided; quantize keeps the strides."""
    sr, gt = _pair(3, 40, 36, 0)
    f = getattr(calculate, f"calculate_{metric}_pt")
    cl = [quantize(t.to(memory_format=torch.channels_last)) for t in (sr, gt)]
    assert not cl[0].is_contiguous()
    assert f(*cl).item() == f(quantize(sr), quantize(gt)).item()


@pytest.mark.parametrize("device", DEVICES)
def test_tensor_psnr_identical_is_inf_and_errors_match(device):
    img = quantize(torch.rand(1, 3, 19, 19)).to(device)
    assert calculate.calculate_psnr_pt(img, img).item() == float("inf")
    assert calculate.calculate_ssim_pt(img, img).item() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="crop_border"):
        calculate.calculate_psnr_pt(img, img, crop_border=-1)
    with pytest.raises(ValueError, match="empty"):
        calculate.calculate_psnr_pt(img[..., :8, :8], img[..., :8, :8])
    with pytest.raises(ValueError, match="11x11"):
        calculate.calculate_ssim_pt(img[..., :16, :16], img[..., :16, :16])
    with pytest.raises(ValueError, match="input_order"):
        calculate.calculate_psnr_pt(img, img, input_order="NHWC")


def _vgg16_random(monkeypatch):
    real = torchvision.models.vgg16
    monkeypatch.setattr(torchvision.models, "vgg16", lambda weights=None: real())


@pytest.mark.parametrize("device", DEVICES)
def test_streaming_dists_is_bit_identical_to_dists_loss(monkeypatch, device):
    """MT-13: per-stage evaluation of both images, same ops and order as dists_loss.forward."""
    from neosr.losses import dists_loss as dl  # noqa: PLC0415

    _vgg16_random(monkeypatch)
    real = dl.dists_loss
    monkeypatch.setattr(
        dl, "dists_loss", lambda as_loss: real(as_loss=as_loss, load_weights=False)
    )
    torch.manual_seed(0)
    m = calculate._streaming_dists().to(device).eval()
    g = torch.Generator().manual_seed(1)
    x, y = (torch.rand(1, 3, 72, 56, generator=g).to(device) for _ in range(2))
    with torch.inference_mode():
        assert torch.equal(m(x, y), m.net(x, y))
        assert m(x, y).ndim == 0


def test_metric_nets_survive_events_and_move_back(monkeypatch):
    """MT-05: one build per process; parked on the CPU between events (VRAM freed), never rebuilt."""
    monkeypatch.setattr(calculate, "_metric_nets", {})
    built = []

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            built.append(1)
            self.w = nn.Parameter(torch.ones(()))

        def forward(self, x, y):
            return (x - y).abs().mean() * self.w

    monkeypatch.setitem(calculate._metric_builders, "topiq", Net)
    dev = torch.device("cuda" if CUDA else "cpu")
    f = build_metric({"type": "calculate_topiq"}, dev)
    img = quantize(torch.rand(1, 1, 8, 8))  # gray: expanded to 3 channels
    for _ in range(2):
        v = f(img, img.flip(-1))
        assert calculate._metric_nets["topiq"].w.device.type == dev.type
        assert v.device.type == dev.type
        calculate.release_metric_nets()
        assert calculate._metric_nets["topiq"].w.device.type == "cpu"
    assert built == [1]


TOPIQ_W = (
    Path(torch.hub.get_dir()) / "checkpoints" / "cfanet_fr_kadid_res50-2c4cc61d.pth"
)


@pytest.mark.skipif(not TOPIQ_W.is_file(), reason="TOPIQ weights not available offline")
def test_topiq_need_weights_false_keeps_the_score(monkeypatch):
    """MT-12 (F22): the attention weights were computed and discarded."""
    monkeypatch.setattr(calculate, "_metric_nets", {})
    g = torch.Generator().manual_seed(0)
    x, y = (quantize(torch.rand(1, 3, 64, 48, generator=g)) for _ in range(2))
    new = calculate.calculate_topiq_pt(x, y, device="cpu").item()
    orig = nn.MultiheadAttention.forward
    monkeypatch.setattr(
        nn.MultiheadAttention,
        "forward",
        lambda self, *a, **k: orig(self, *a, **(k | {"need_weights": True})),
    )
    old = calculate.calculate_topiq_pt(x, y, device="cpu").item()
    assert abs(new - old) <= 1e-6


# ---------------------------------------------------------------------------- validation
S = 2


class Tiny(nn.Module):
    """Exact under tiling for halo >= 2 (RF 2)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3 * S * S, 3, padding=1)
        self.conv2 = nn.Conv2d(3 * S * S, 3 * S * S, 3, padding=1)
        self.shuffle = nn.PixelShuffle(S)
        self.calls: list[tuple[int, ...]] = []
        self.dtypes: list[torch.dtype] = []

    def forward(self, x):
        self.calls.append(tuple(x.shape))
        h = self.conv(x)
        self.dtypes.append(h.dtype)
        return self.shuffle(self.conv2(h)) * 0.5 + 0.25


class PairSet(Dataset):
    def __init__(
        self, files, name="valset", root="/data/lq", kind="paired", hw=(12, 10)
    ):
        self.opt = {"name": name, "type": kind, "dataroot_lq": root}
        g = torch.Generator().manual_seed(3)
        h, w = hw
        self.items = [
            (
                f,
                torch.rand(3, h, w, generator=g),
                torch.rand(3, h * S, w * S, generator=g),
            )
            for f in files
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        f, lq, gt = self.items[i]
        out = {"lq": lq, "lq_path": f}
        return out if self.opt["type"] == "single" else out | {"gt": gt}


def make_model(tmp, *, is_train=False, metrics=None, val=None, device=None, **top):
    opt = {
        "name": "tiny",
        "model_type": "image",
        "scale": S,
        "is_train": False,
        "network_g": {"type": "compact", "num_feat": 4, "num_conv": 1},
        "datasets": {},
        "val": {"pbar": False, "metrics": metrics or {}} | (val or {}),
        "logger": {},
        "path": {"visualization": tmp / "vis"},
    } | top
    rt = CPU if device is None else Runtime(torch.device(device), 0, 1, 0, False)
    m = build_model(opt, rt)
    m.net_g = Tiny().to(rt.device)
    m.is_train = is_train  # the validate contract only; no phases needed
    return m


ALL_METRICS = {
    "psnr": {"type": "calculate_psnr", "crop_border": 2},
    "ssim": {"type": "calculate_ssim", "crop_border": 0},
    "psnr_y": {"type": "calculate_psnr", "crop_border": 0, "test_y_channel": True},
    "ssim_y": {"type": "calculate_ssim", "crop_border": 2, "test_y_channel": True},
}


@pytest.mark.parametrize("device", DEVICES)
def test_validation_metrics_equal_the_numpy_route(tmp_path, device):
    m = make_model(tmp_path, metrics=ALL_METRICS, device=device)
    ds = PairSet([f"/data/lq/im{i}.png" for i in range(3)], hw=(14, 12))
    m.validation(DataLoader(ds, batch_size=1), "x", None, save_img=False)
    net = m.net_g.eval()
    for k, cfg in ALL_METRICS.items():
        kw = {a: b for a, b in cfg.items() if a != "type"}
        vals = []
        for _, lq, gt in ds.items:
            with torch.inference_mode():
                sr = net(lq[None].to(device))
            f = getattr(calculate, cfg["type"])
            vals.append(float(f(tensor2img([sr]), tensor2img([gt[None]]), **kw)))
        got = m.best_metric_results["valset"][k]["val"]
        assert abs(got - sum(vals) / len(vals)) <= 1e-10


def test_validation_skips_conversion_nothing_consumes(tmp_path, monkeypatch):
    m = make_model(tmp_path)
    calls = []
    monkeypatch.setattr(image_mod, "quantize", lambda t: calls.append(1) or quantize(t))
    ds = PairSet(["/data/lq/a.png"])
    m.validation(DataLoader(ds, batch_size=1), "x", None, save_img=False)
    assert calls == []
    assert len(m.net_g.calls) == 1
    m.validation(DataLoader(ds, batch_size=1), "x", None, save_img=True)
    assert calls == [1]


def test_validation_runs_without_cudnn_autotune(tmp_path):
    seen = []
    m = make_model(tmp_path, metrics={"p": {"type": "calculate_psnr"}})
    m.metrics = {
        "p": lambda *_: seen.append(torch.backends.cudnn.benchmark) or torch.zeros(())
    }
    m.opt["val"]["metrics"] = {"p": {"better": "higher"}}
    was = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = True
    try:
        m.validation(
            DataLoader(PairSet(["/data/lq/a.png"]), batch_size=1), 1, None, False
        )
        assert seen == [False]
        assert torch.backends.cudnn.benchmark is True  # training flags restored
    finally:
        torch.backends.cudnn.benchmark = was


class OOMOnce(Tiny):
    """Raises a CUDA OOM for full images (wider than `limit`) `n` times."""

    def __init__(self, limit, n=1) -> None:
        super().__init__()
        self.limit, self.n = limit, n

    def forward(self, x):
        if x.shape[-1] > self.limit and self.n:
            self.n -= 1
            msg = "CUDA out of memory (test)"
            raise torch.OutOfMemoryError(msg)
        return super().forward(x)


def test_validation_oom_retries_tiled_and_warns_once(tmp_path, caplog):
    m = make_model(
        tmp_path, metrics={"psnr": {"type": "calculate_psnr", "crop_border": 0}}
    )
    ds = PairSet([f"/data/lq/im{i}.png" for i in range(2)], hw=(20, 300))
    ref = make_model(tmp_path / "ref", metrics=m.opt["val"]["metrics"])
    ref.net_g.load_state_dict(m.net_g.state_dict())
    ref.validation(DataLoader(ds, batch_size=1), 1, None, save_img=False)
    net = OOMOnce(limit=256, n=2)
    net.load_state_dict(m.net_g.state_dict())
    m.net_g, m.tile_overlap = net, 4
    with caplog.at_level(logging.WARNING, logger="neosr"):
        m.validation(DataLoader(ds, batch_size=1), 1, None, save_img=False)
    warns = [r for r in caplog.records if "out of memory" in r.getMessage()]
    assert len(warns) == 1
    assert "tile 256" in warns[0].getMessage()
    assert max(c[-1] for c in net.calls) == 256  # every image fell back to 256 tiles
    # RF 2 <= halo 4: the tiled output is the full one
    got = m.best_metric_results["valset"]["psnr"]["val"]
    assert got == pytest.approx(
        ref.best_metric_results["valset"]["psnr"]["val"], abs=1e-9
    )


def test_validation_second_oom_propagates(tmp_path):
    m = make_model(tmp_path)
    m.net_g = OOMOnce(limit=0, n=2)  # the tiled retry fails too
    with pytest.raises(torch.OutOfMemoryError):
        m.validation(
            DataLoader(PairSet(["/data/lq/a.png"]), batch_size=1), 1, None, False
        )


class Watch(Tiny):
    """Records, at every forward, how many of the watched tensors are still alive."""

    def __init__(self, refs) -> None:
        super().__init__()
        self.refs, self.alive = refs, []

    def forward(self, x):
        self.alive.append(sum(r() is not None for r in self.refs))
        return super().forward(x)


def test_validation_frees_the_image_buffers_before_the_next_forward(tmp_path):
    """Review: the quantised sr and gt of one image must not live into the next forward."""
    refs = []
    m = make_model(tmp_path, metrics={"p": {"type": "calculate_psnr"}})
    m.metrics = {
        "p": lambda q, gt: refs.extend(map(weakref.ref, (q, gt))) or torch.zeros(())
    }
    m.net_g = Watch(refs)
    m.validation(DataLoader(PairSet(FILES), batch_size=1), 1, None, save_img=True)
    assert len(refs) == 6
    assert m.net_g.alive == [0, 0, 0]


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_tiled_output_metrics_run_on_the_cpu_and_mix_with_device_ones(tmp_path):
    """Review: a tiled output (here the OOM retry of the first image) is on the CPU; its PSNR/SSIM
    stay there (full-size fp64 statistics), the untiled ones run on the device, and the event
    still averages both."""
    cfg = {"psnr": ALL_METRICS["psnr"], "ssim": ALL_METRICS["ssim"]}
    ds = PairSet([f"/data/lq/im{i}.png" for i in range(2)], hw=(20, 300))
    ref = make_model(tmp_path / "ref", metrics=cfg, device="cuda")
    ref.validation(DataLoader(ds, batch_size=1), 1, None, save_img=False)
    m = make_model(tmp_path, metrics=cfg, device="cuda")
    m.net_g, m.tile_overlap = OOMOnce(limit=256, n=1).cuda(), 4
    m.net_g.load_state_dict(ref.net_g.state_dict())
    seen = []
    for k, f in list(m.metrics.items()):
        m.metrics[k] = lambda q, gt, f=f: (
            seen.append((q.device.type, gt.device.type)) or f(q, gt)
        )
    m.validation(DataLoader(ds, batch_size=1), 1, None, save_img=False)
    assert seen == [("cpu", "cpu")] * 2 + [("cuda", "cuda")] * 2
    for k in cfg:
        assert m.best_metric_results["valset"][k]["val"] == pytest.approx(
            ref.best_metric_results["valset"][k]["val"], abs=1e-9
        )


def test_a_test_run_named_lq_keeps_outputs_and_lq_copies_apart(tmp_path):
    """Review: the run name is the test output tag; "lq" must not route outputs to the LQ copies."""
    m = make_model(tmp_path, name="lq")
    m.validation(DataLoader(PairSet(FILES[:1]), batch_size=1), "lq", None, True)
    vis = tmp_path / "vis"
    got = sorted(p.relative_to(vis).as_posix() for p in vis.rglob("*.png"))
    assert got == ["lq/valset/a/x.png", "valset/a/x_lq.png"]
    want = tensor2img([m.net_g.eval()(PairSet(FILES[:1]).items[0][1][None]).detach()])
    np.testing.assert_array_equal(cv2.imread(str(vis / "valset/a/x_lq.png")), want)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_device_metrics_do_not_sync_the_host():
    """Review: the per-image metric calls queue work only; the event syncs once at the end."""
    g = torch.Generator().manual_seed(0)
    x, y = (quantize(torch.rand(1, 3, 32, 24, generator=g).cuda()) for _ in range(2))
    calls = [
        lambda: calculate.calculate_psnr_pt(x, y, 2, test_y_channel=True),
        lambda: calculate.calculate_ssim_pt(x, y, 0),
        lambda: calculate.calculate_ssim_pt(x, y, 2, test_y_channel=True),
    ]
    for f in calls:  # first use: the kernel is copied to the device once
        f()
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for f in calls:
            f()
    finally:
        torch.cuda.set_sync_debug_mode(0)


class TB:
    def __init__(self) -> None:
        self.images, self.scalars = [], []

    def add_image(self, tag, img, global_step, dataformats):
        self.images.append((tag, img.shape, global_step, dataformats))

    def add_scalar(self, *a):
        self.scalars.append(a)


FILES = ["/data/lq/a/x.png", "/data/lq/b/x.png", "/data/lq/x.png"]


def test_training_outputs_keep_datasets_and_subfolders_apart(tmp_path):
    m = make_model(tmp_path, is_train=True, logger={"save_tb_img": True})
    tb = TB()
    for name in ("set1", "set2"):
        m.validation(DataLoader(PairSet(FILES, name=name), batch_size=1), 7, tb, True)
    vis = tmp_path / "vis"
    got = sorted(p.relative_to(vis).as_posix() for p in vis.rglob("*.png"))
    want = sorted(
        f"{ds}/{rel}/{Path(rel).name}_{tag}.png"
        for ds in ("set1", "set2")
        for rel in ("a/x", "b/x", "x")
        for tag in (7, "lq")
    )
    assert got == want
    # TB: one tag per image (the slider is the step), dataset-qualified
    assert [t[0] for t in tb.images] == [
        f"{ds}/{r}" for ds in ("set1", "set2") for r in ("a/x", "b/x", "x")
    ]
    assert {t[2] for t in tb.images} == {7}
    assert {t[1] for t in tb.images} == {(12 * S, 10 * S, 3)}


def test_lq_is_saved_per_source_and_outputs_match_tensor2img(tmp_path):
    m = make_model(tmp_path)
    ds = PairSet(FILES)
    m.validation(DataLoader(ds, batch_size=1), "x", None, True)
    vis = tmp_path / "vis"
    got = sorted(p.relative_to(vis).as_posix() for p in vis.rglob("*.png"))
    assert got == sorted(
        ["valset/a/x_tiny.png", "valset/b/x_tiny.png", "valset/x_tiny.png"]
        + ["lq/valset/a/x.png", "lq/valset/b/x.png", "lq/valset/x.png"]
    )
    net = m.net_g.eval()
    for (f, lq, _), rel in zip(ds.items, ("a/x", "b/x", "x"), strict=True):
        with torch.inference_mode():
            want = tensor2img([net(lq[None])])
        out = cv2.imread(str(vis / "valset" / f"{rel}_tiny.png"))
        np.testing.assert_array_equal(out, want)
        lq_out = cv2.imread(str(vis / "lq" / "valset" / f"{rel}.png"))
        np.testing.assert_array_equal(lq_out, tensor2img([lq[None]]))
        assert Path(f).name == "x.png"


def test_outside_dataroot_and_lmdb_keys_fall_back_to_the_stem():
    assert image_mod.image_id("/data/lq/a/x.png", "/data/lq") == Path("a/x")
    assert image_mod.image_id("/elsewhere/x.png", "/data/lq") == Path("x")
    assert image_mod.image_id("0001", "/data/lq.lmdb") == Path("0001")
    assert image_mod.image_id("/data/lq/x.y.png", None) == Path("x.y")


def test_images_are_written_by_the_writer_thread(tmp_path, monkeypatch):
    threads = []
    real = image_mod.imwrite
    monkeypatch.setattr(
        image_mod,
        "imwrite",
        lambda img, p: threads.append(threading.current_thread().name) or real(img, p),
    )
    m = make_model(tmp_path)
    m.validation(DataLoader(PairSet(FILES), batch_size=1), "x", None, True)
    assert len(threads) == 6
    assert all(t.startswith("val_imwrite") for t in threads)
    assert len(list((tmp_path / "vis").rglob("*.png"))) == 6  # all written on return


def test_a_failed_write_raises(tmp_path, monkeypatch):
    def fail(*_):
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(image_mod, "imwrite", fail)
    m = make_model(tmp_path)
    with pytest.raises(OSError, match="disk full"):
        m.validation(DataLoader(PairSet(FILES), batch_size=1), "x", None, True)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize(
    ("is_train", "amp", "bf16", "want"),
    [
        (False, True, True, torch.bfloat16),
        (False, True, False, torch.float16),
        (False, False, True, torch.float32),
        (True, True, True, torch.float32),  # training validation stays fp32 (GP-13)
    ],
)
def test_inference_autocast_is_opt_in(tmp_path, is_train, amp, bf16, want):
    m = make_model(
        tmp_path,
        is_train=is_train,
        device="cuda",
        metrics={"psnr": {"type": "calculate_psnr", "crop_border": 0}},
        use_amp=amp,
        bfloat16=bf16,
    )
    m.validation(DataLoader(PairSet(FILES[:1]), batch_size=1), "x", None, False)
    assert m.net_g.dtypes == [want]
    assert math.isfinite(m.best_metric_results["valset"]["psnr"]["val"])


# ---------------------------------------------------------------------------- convert.py
def _convert():
    import importlib.util  # noqa: PLC0415

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("convert_py", root / "convert.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


convert_py = _convert()


def _args(tmp, *extra, net="compact"):
    from neosr.cli import convert_args  # noqa: PLC0415

    return convert_args(
        ["--input", str(tmp / "w.pth"), "-net", net, "-s", "2"]
        + ["--output", str(tmp / "m.onnx"), *extra]
    )


class FrozenW(nn.Module):
    """A trace-time constant from the width, like DySample's normalizer (F07)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x):
        return self.conv(x) * torch.tensor(float(x.shape[-1])) / 20


class FrozenB(FrozenW):
    """A trace-time constant from the batch, like window_reverse's int(B) (F08)."""

    def forward(self, x):
        return self.conv(x) + x.sum(0, keepdim=True) / int(x.shape[0])


class BatchOne(FrozenW):
    def forward(self, x):
        if x.shape[0] != 1:
            msg = "batch 1 only"
            raise ValueError(msg)
        return self.conv(x)


@pytest.fixture
def onnx_deps():
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")


@pytest.mark.parametrize(
    ("net", "at"), [(FrozenW, "1x3x32x24"), (FrozenB, "2x3x20x20")]
)
def test_dynamic_export_wrong_off_the_trace_shape_fails(tmp_path, onnx_deps, net, at):
    """MT-02: the trace shape passes; the other dynamic shapes expose the frozen constant."""
    with pytest.raises(RuntimeError, match=f"wrong at {at} .*use -static C H W"):
        convert_py.to_onnx(_args(tmp_path), net())
    # static: verified at its own shape only, and correct there
    convert_py.to_onnx(_args(tmp_path, "-static", "3", "20", "20"), net())


def test_shape_the_torch_model_rejects_is_skipped(tmp_path, onnx_deps, caplog):
    with caplog.at_level(logging.WARNING, logger="neosr"):
        convert_py.to_onnx(_args(tmp_path), BatchOne())
    assert any("skips 2x3x20x20" in r.getMessage() for r in caplog.records)


def test_dynamic_dim_names_are_distinct(tmp_path, onnx_deps):
    """MT-07: one dim_param string means "equal sizes" to the runtime; NCHW dim 2 is the height."""
    import onnx  # noqa: PLC0415

    convert_py.to_onnx(_args(tmp_path), nn.Sequential(nn.Conv2d(3, 3, 3, padding=1)))
    graph = onnx.load(str(tmp_path / "m_fp32.onnx")).graph
    dims = [
        [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]
        for v in (graph.input[0], graph.output[0])
    ]
    assert dims == [
        ["batch", 3, "height", "width"],
        ["batch", 3, "out_height", "out_width"],
    ]


def test_fp16_export_is_float16_and_verified(tmp_path, onnx_deps):
    """MT-16: a real fp16 graph (fp16 weights and I/O), verified at the MT-02 shapes."""
    import onnx  # noqa: PLC0415

    from neosr.archs import build_network  # noqa: PLC0415

    torch.manual_seed(0)
    sd = build_network({"type": "compact"}, scale=2).state_dict()
    torch.save({"params": sd}, tmp_path / "w.pth")
    argv = ["--input", str(tmp_path / "w.pth"), "-net", "compact", "-s", "2", "-fp16"]
    argv += ["--output", str(tmp_path / "m.onnx")]
    assert convert_py.main(argv) == 0
    m = onnx.load(str(tmp_path / "m_fp16.onnx"))
    f16 = onnx.TensorProto.FLOAT16
    assert m.graph.input[0].type.tensor_type.elem_type == f16
    assert m.graph.output[0].type.tensor_type.elem_type == f16
    assert {i.data_type for i in m.graph.initializer if len(i.dims) > 1} == {f16}
    assert (tmp_path / "m_fp32.onnx").is_file()


def test_fp16_without_the_converter_fails_before_writing(
    tmp_path, onnx_deps, monkeypatch, capsys
):
    import sys  # noqa: PLC0415

    monkeypatch.setitem(sys.modules, "onnxruntime.transformers.float16", None)
    argv = ["--input", str(tmp_path / "w.pth"), "-net", "compact", "-fp16"]
    assert convert_py.main(argv + ["--output", str(tmp_path / "m.onnx")]) == 2
    assert "float16" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (["-net", "atd"], "atd.*cannot be exported.*Sort"),
        (["-net", "esc"], "esc.*cannot be exported.*dynamic conv"),
        (["-net", "safmn"], "static shape.*-static C H W"),
        (["-net", "srformer_medium"], "static shape.*shift mask"),
        (["-net", "compact", "--fulloptimization"], "fulloptimization"),
    ],
)
def test_export_capability_is_checked_up_front(
    tmp_path, onnx_deps, capsys, extra, match
):
    """MT-17: a clear config error (exit 2) before the weights are read or a file is written."""
    argv = [
        "--input",
        str(tmp_path / "missing.pth"),
        "--output",
        str(tmp_path / "m.onnx"),
    ]
    assert convert_py.main(argv + extra) == 2
    import re  # noqa: PLC0415

    assert re.search(match, capsys.readouterr().err)
    assert not list(tmp_path.iterdir())


def test_static_only_arch_exports_with_static(tmp_path, onnx_deps):
    convert_py.check_export(_args(tmp_path, "-static", "3", "16", "16", net="safmn"))


@pytest.mark.parametrize(
    ("net", "static"), [("craft", None), ("hit_srf", None), ("ditn", ["3", "20", "20"])]
)
def test_arch_exports_and_verifies_after_the_stage2_fixes(
    tmp_path, onnx_deps, net, static
):
    """MT-17 re-sweep: archs the stage-2 lanes made exportable leave NO_ONNX and verify."""
    from neosr.archs import build_network  # noqa: PLC0415

    torch.manual_seed(0)
    extra = ["-static", *static] if static else []
    args = _args(tmp_path, *extra, net=net)
    convert_py.to_onnx(args, build_network({"type": net}, scale=2))


@pytest.mark.parametrize("net", ["asid", "omnisr", "ditn", "dctlsa", "microsr_light"])
def test_arch_exports_dynamically_after_the_export_fixes(tmp_path, onnx_deps, net):
    """Export sweep fixes (ceil-div padding, traced shapes, window_reverse batch): the dynamic
    graph verifies at every export_shapes() input, and the arch is not refused up front."""
    from neosr.archs import build_network  # noqa: PLC0415

    assert net not in convert_py.NO_ONNX
    assert net not in convert_py.STATIC_ONLY
    torch.manual_seed(0)
    convert_py.to_onnx(_args(tmp_path, net=net), build_network({"type": net}, scale=2))


class QFrozenB(nn.Module):
    """A QDQ graph with a trace-time batch constant (window_reverse's int(B), F08)."""

    def __init__(self) -> None:
        super().__init__()
        from torch.ao.quantization import FakeQuantize  # noqa: PLC0415

        self.fq = FakeQuantize()
        self.conv = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x):
        x = self.fq(x)
        return self.conv(x) + x.sum(0, keepdim=True) / int(x.shape[0])


def test_qat_export_is_verified_at_every_shape(tmp_path, onnx_deps):
    """Review/MT-02: the QDQ check runs at the export_shapes() inputs too, not only at batch 1."""
    net = QFrozenB()
    net(torch.rand(2, 3, 20, 20))  # calibrate, then freeze the ranges
    net.fq.disable_observer()
    assert convert_py.is_qat(net)
    with pytest.raises(RuntimeError, match="deviates .* at 2x3x20x20"):
        convert_py.to_onnx(_args(tmp_path), net)
    convert_py.to_onnx(_args(tmp_path, "-static", "3", "20", "20"), net)


@pytest.mark.parametrize("dep", ["onnx", "onnxruntime"])
def test_missing_onnx_dependency_is_a_config_error(tmp_path, monkeypatch, capsys, dep):
    """Review/MT-17: exit 2 with a message, before the weights are read or a file is written."""
    import sys  # noqa: PLC0415

    monkeypatch.setitem(sys.modules, dep, None)
    argv = ["--input", str(tmp_path / "missing.pth"), "-net", "compact"]
    assert convert_py.main(argv + ["--output", str(tmp_path / "m.onnx")]) == 2
    assert "needs onnx and onnxruntime" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())
