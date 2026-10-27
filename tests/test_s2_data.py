"""Stage 2, data lane (docs/refactor/stage2/analysis/data.md): D-01 .. D-20."""

import logging
import os
import pickle
import random
from functools import partial
from itertools import starmap
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from neosr.config import ConfigError
from neosr.data import build_dataset, otf_dataset, paired_dataset, worker_init_fn
from neosr.data import data_util as du
from neosr.data.augmentations import resizemix
from neosr.data.otf_degradation import Degrader, PairPool
from neosr.utils import misc
from neosr.utils.rng import seed_rng

KERNELS = {
    "kernel_list": ["iso", "aniso", "generalized_iso", "plateau_aniso"],
    "kernel_prob": [0.4, 0.3, 0.2, 0.1],
    "blur_sigma": [0.2, 3],
    "betag_range": [0.5, 4],
    "betap_range": [1, 2],
}
OTF_KEYS = {
    **KERNELS,
    **{f"{k}2": v for k, v in KERNELS.items()},
    "sinc_prob": 0.3,
    "sinc_prob2": 0.3,
    "final_sinc_prob": 0.5,
}
DEG = {
    "resize_prob": [0.3, 0.4, 0.3], "resize_range": [0.5, 1.5], "gaussian_noise_prob": 0.5,
    "noise_range": [0, 2], "poisson_scale_range": [0.05, 0.25], "gray_noise_prob": 0.5,
    "jpeg_range": [40, 95], "second_blur_prob": 0.5, "resize_prob2": [0.3, 0.4, 0.3],
    "resize_range2": [0.3, 1.5], "gaussian_noise_prob2": 0.5, "noise_range2": [0, 2],
    "poisson_scale_range2": [0.05, 0.1], "gray_noise_prob2": 0.5, "jpeg_range2": [35, 95],
}  # fmt: skip
CUDA = torch.cuda.is_available()


def _seed(n: int) -> None:
    random.seed(n)
    seed_rng(n)
    torch.manual_seed(n)


def _img(h: int, w: int, seed: int, depth: type = np.uint8) -> np.ndarray:
    hi = np.iinfo(depth).max + 1
    return np.random.default_rng(seed).integers(0, hi, (h, w, 3), dtype=depth)


@pytest.fixture
def pairs(tmp_path: Path) -> Path:
    """Non-square pairs, x4: 8-bit and 16-bit PNGs (IMREAD_COLOR decodes both to uint8)."""
    for sub in ("GT", "LQ"):
        (tmp_path / sub).mkdir()
    for i, depth in enumerate((np.uint8, np.uint16, np.uint8)):
        gt = _img(200, 296, i, depth)
        cv2.imwrite(str(tmp_path / "GT" / f"p{i}.png"), gt)
        cv2.imwrite(str(tmp_path / "LQ" / f"p{i}.png"), gt[::4, ::4])
    return tmp_path


def _paired_opt(root: Path, phase: str = "train", **kw) -> dict:
    return {"type": "paired", "phase": phase, "dataroot_gt": str(root / "GT"),
            "dataroot_lq": str(root / "LQ"), "scale": 4, "patch_size": 24, **kw}  # fmt: skip


def _otf_opt(root: Path, patch: int = 24, **kw) -> dict:
    return {"type": "otf", "phase": "train", "dataroot_gt": str(root / "GT"), "scale": 4,
            "patch_size": patch, **OTF_KEYS, **kw}  # fmt: skip


def _float_first(monkeypatch, module) -> None:
    """The pre-D-06 order: the whole image is converted to float32 before the crop."""
    monkeypatch.setattr(module, "read_img", partial(_read_float, du.read_img))
    monkeypatch.setattr(module, "to_float32", lambda x: x)


def _read_float(read, client, path, key, *_, **__):
    return read(client, path, key)


def _items(ds, n: int = 6) -> list[dict]:
    _seed(3)
    return [ds[i % len(ds)] for i in range(n)]


def _assert_items_equal(a: list[dict], b: list[dict]) -> None:
    for x, y in zip(a, b, strict=True):
        assert x.keys() == y.keys()
        for k in x:
            if torch.is_tensor(x[k]):
                assert x[k].dtype == y[k].dtype
                assert torch.equal(x[k], y[k]), k
            else:
                assert x[k] == y[k]


# D-06 (a): crop in uint8, then convert: identical to converting first
@pytest.mark.parametrize("phase", ["train", "val"])
@pytest.mark.parametrize("color", [None, "y"])
def test_paired_crop_before_float_is_exact(pairs, monkeypatch, phase, color):
    ds = build_dataset(_paired_opt(pairs, phase, color=color))
    new = _items(ds)
    _float_first(monkeypatch, paired_dataset)
    old = _items(build_dataset(_paired_opt(pairs, phase, color=color)))
    _assert_items_equal(new, old)
    assert new[0]["gt"].shape[0] == (1 if color else 3)


@pytest.mark.parametrize("patch", [24, 96])  # canvas 128 (crop) and 416 (pad) of a 200x296 GT
def test_otf_crop_before_float_is_exact(pairs, monkeypatch, patch):
    new = _items(build_dataset(_otf_opt(pairs, patch)))
    _float_first(monkeypatch, otf_dataset)
    old = _items(build_dataset(_otf_opt(pairs, patch)))
    _assert_items_equal(new, old)
    assert new[0]["gt"].shape == (3, *(otf_dataset.otf_crop_size(patch, 4),) * 2)


# D-01: bounded training retry
class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.msgs: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.msgs.append(record.getMessage())


@pytest.fixture
def warnings_seen():
    h = _Capture()
    log = logging.getLogger("neosr")
    log.addHandler(h)
    level = log.level
    log.setLevel(logging.INFO)
    yield h.msgs
    log.removeHandler(h)
    log.setLevel(level)


def test_retry_skips_bad_and_mis_sized(pairs, warnings_seen):
    (pairs / "GT" / "p1.png").write_bytes(b"")  # unreadable
    cv2.imwrite(str(pairs / "LQ" / "p2.png"), _img(10, 10, 0))  # wrong scale
    ds = build_dataset(_paired_opt(pairs))
    _seed(0)
    for i in range(3):
        assert Path(ds[i]["gt_path"]).name == "p0.png"
    assert any("p1.png" in m for m in warnings_seen)
    assert any("p2.png" in m and "Scale mismatches" in m for m in warnings_seen)
    with pytest.raises(OSError, match=r"p1\.png"):
        build_dataset(_paired_opt(pairs, "val"))[1]


def test_retry_bounded(pairs, warnings_seen):
    for f in (pairs / "GT").iterdir():
        f.write_bytes(b"broken")
    ds = build_dataset(_otf_opt(pairs))
    with pytest.raises(RuntimeError, match="10 consecutive") as e:
        ds[0]
    assert isinstance(e.value.__cause__, OSError)
    assert len(warnings_seen) == 10


# D-02: canvas key
def test_otf_legacy_canvas_key(pairs):
    assert build_dataset(_otf_opt(pairs, 64)).crop_pad_size == 288
    assert build_dataset(_otf_opt(pairs, 64, legacy_canvas=True)).crop_pad_size == 512


# D-09 / F15, D-10 / F16
def _tree(root: Path) -> None:
    for d in ("GT/sub", "GT/.thumbs", "LQ/sub", "LQ/.thumbs"):
        (root / d).mkdir(parents=True)
    for side in ("GT", "LQ"):
        for name in ("b.png", "a.PNG", "sub/c.jpg", ".thumbs/x.png"):
            cv2.imwrite(str(root / side / name), np.zeros((4, 4, 3), np.uint8))
        for junk in (".DS_Store", "Thumbs.db", "desktop.ini", "notes.txt"):
            (root / side / junk).write_bytes(b"x")


def test_scandir_skips_hidden_and_non_images(tmp_path):
    _tree(tmp_path)
    (tmp_path / "GT" / "link").symlink_to(tmp_path / "GT" / "sub", target_is_directory=True)
    names = [Path(p).relative_to(tmp_path / "GT").as_posix()
             for p in du.image_paths(str(tmp_path / "GT"), recursive=True)]  # fmt: skip
    assert names == ["a.PNG", "b.png", "sub/c.jpg"]  # sorted; no symlinked folder
    assert du.image_paths(str(tmp_path / "GT")) == [
        str(tmp_path / "GT" / n) for n in ("a.PNG", "b.png")
    ]
    assert sorted(misc.scandir(str(tmp_path / "GT"), suffix="png")) == ["a.PNG", "b.png"]
    otf = build_dataset({**_otf_opt(tmp_path), "patch_size": 1})
    assert [Path(p).name for p in otf.paths] == ["a.PNG", "b.png"]


def test_folder_pairing_sorted(tmp_path, monkeypatch):
    _tree(tmp_path)
    folders, keys = [str(tmp_path / "LQ"), str(tmp_path / "GT")], ["lq", "gt"]
    ref = du.paired_paths_from_folder(folders, keys)
    scan = os.scandir
    monkeypatch.setattr(misc.os, "scandir", lambda p: iter(sorted(scan(p), key=lambda e: e.name)[::-1]))
    assert du.paired_paths_from_folder(folders, keys) == ref
    assert [Path(p["gt_path"]).name for p in ref] == ["a.PNG", "b.png", "c.jpg"]


# D-07: meta_info names with spaces and dots
def test_meta_info_names(tmp_path):
    meta = tmp_path / "meta.txt"
    meta.write_text(
        "my photo.png (256,256,3) 1\na.b.png (8,8,3)\nplain.png\n\nü x.png (1,2,3) 0\r\n",
        encoding="utf-8",
    )
    assert du.meta_info_names(str(meta)) == ["my photo.png", "a.b.png", "plain.png", "ü x.png"]


# D-18: compact path lists
def test_path_list():
    paths = ["a/b.png", "ü/ç d.png", "", os.fsdecode(b"bad\xff.png")]
    pl = du.PathList(paths)
    assert len(pl) == 4
    assert list(pl) == paths
    assert pl[-1] == paths[-1]
    assert pl[1] == paths[1]
    with pytest.raises(IndexError):
        pl[4]
    assert list(pickle.loads(pickle.dumps(pl))) == paths  # spawn workers pickle the dataset
    assert list(du.PathList([])) == []


# D-19
def test_numpy_errstate_not_global():
    from neosr.data import degradations  # ruff: ignore[import-outside-top-level]

    assert np.geterr()["divide"] == "warn"
    with np.errstate(divide="raise", invalid="raise"):
        k = degradations.circular_lowpass_kernel(np.pi / 2, 13, pad_to=21)
    assert k.shape == (21, 21)
    assert np.isfinite(k).all()


@pytest.mark.parametrize(
    ("change", "key"),
    [
        ({"sinc_prob": None}, "sinc_prob"),
        ({"final_sinc_prob": "0.5"}, "final_sinc_prob"),
        ({"kernel_list2": ["iso", "gauss"]}, "kernel_list2"),
        ({"kernel_prob": [0.5, 0.5]}, "kernel_prob"),
        ({"betag_range": None}, "betag_range"),  # generalized_iso is listed
        ({"blur_sigma2": [1]}, "blur_sigma2"),
        ({"blur_sigmay": [1, "2"]}, "blur_sigmay"),
    ],
)
def test_otf_kernel_keys_validated(pairs, change, key):
    with pytest.raises(ConfigError, match=key):
        build_dataset({**_otf_opt(pairs), **change})


def test_otf_kernel_keys_optional_ranges(pairs):
    opt = {**_otf_opt(pairs), "kernel_list": ["iso"], "kernel_prob": [1], "betag_range": None}
    build_dataset(opt)[0]


# D-04: resizemix targets without aliasing
def test_resizemix_antialiased():
    """Downscaled white noise: antialiasing averages it out (std 0.06 at 4x), plain bicubic
    samples it (aliasing, std 0.21)."""
    gt = torch.rand(2, 3, 64, 64, generator=torch.Generator().manual_seed(0))
    seed_rng(0)
    out, _ = resizemix(gt.clone(), gt[..., ::2, ::2].clone(), scope=(0.25, 0.25), scale=2)
    box = (out != gt).all(1)
    assert box.sum(dim=(1, 2)).min() >= 8 * 8  # the 16x16 box, at most halved by the border
    assert out[:, 0][box].std() < 0.1


# D-05: no host sync in the noise path
@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_degrader_no_host_sync():
    torch.manual_seed(0)
    d = Degrader(DEG, scale=4, device="cuda")
    gt = torch.rand(4, 3, 96, 96, device="cuda")
    ks = [torch.full((4, n, n), 1 / n**2, device="cuda") for n in (7, 9, 21)]
    d(gt, *ks)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for i in range(20):
            _seed(i)
            d(gt, *ks)
    finally:
        torch.cuda.set_sync_debug_mode(0)


# D-11: grayscale OTF
@pytest.mark.parametrize("jpeg", [True, False])
def test_degrade_gray(jpeg):
    cfg = {**DEG, "color": "y"} if jpeg else {
        **{k: v for k, v in DEG.items() if not k.startswith("jpeg")}, "color": "y", "jpeg_compress": False}  # fmt: skip
    d = Degrader(cfg, scale=4, device="cpu")
    gt = torch.rand(2, 3, 64, 64, generator=torch.Generator().manual_seed(0))
    ks = [torch.full((2, n, n), 1 / n**2) for n in (7, 9, 21)]
    _seed(0)
    gy, ly = d.degrade(gt, *ks)
    _seed(0)
    lq_rgb = d(gt, *ks)
    assert gy.shape == (2, 1, 64, 64)
    assert ly.shape == (2, 1, 16, 16)
    assert torch.equal((ly * 255).round() / 255, ly)  # re-quantised
    # luma as the paired datasets compute it (cv2)
    ref = cv2.cvtColor(gt[0].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY)
    torch.testing.assert_close(gy[0, 0], torch.from_numpy(ref), atol=1e-6, rtol=0)
    ref_lq = cv2.cvtColor(lq_rgb[0].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY)
    assert (ly[0, 0] - torch.from_numpy(ref_lq)).abs().max() <= 0.5 / 255 + 1e-6
    rgb = Degrader({k: v for k, v in cfg.items() if k != "color"}, scale=4, device="cpu")
    _seed(0)
    g, lq = rgb.degrade(gt, *ks)
    assert g is gt
    assert torch.equal(lq, lq_rgb)


def test_otf_dataset_gray_stays_rgb(pairs):
    assert build_dataset(_otf_opt(pairs, color="y"))[0]["gt"].shape[0] == 3


# D-15: uint8 pair pool, bit-exact against the fp32 pool
class _OldPool:
    """PairPool at f3d2b7c (fp32 storage)."""

    def __init__(self, size, batch):
        self.batch, self.size, self.ptr = batch, size // batch * batch, 0
        self.lq = self.gt = None

    def __call__(self, lq, gt):
        b = self.batch
        if self.lq is None:
            self.lq = lq.new_empty((self.size, *lq.shape[1:]))
            self.gt = gt.new_empty((self.size, *gt.shape[1:]))
        if self.ptr < self.size:
            self.lq[self.ptr : self.ptr + b] = lq
            self.gt[self.ptr : self.ptr + b] = gt
            self.ptr += b
            return lq, gt
        idx = torch.randperm(self.size)[:b]
        out_lq, out_gt = self.lq[idx], self.gt[idx]
        self.lq[idx], self.gt[idx] = lq, gt
        return out_lq, out_gt


@pytest.mark.parametrize("gt_exact", [True, False])
def test_pair_pool_uint8_bit_exact(gt_exact):
    gen = torch.Generator().manual_seed(0)

    def batch():
        lq = (torch.rand(3, 3, 4, 4, generator=gen) * 255).round() / 255  # Degrader output
        gt = torch.rand(3, 3, 16, 16, generator=gen)
        return lq, (gt * 255).round() / 255 if gt_exact else gt

    batches = [batch() for _ in range(12)]
    new, old = PairPool(9, 3, device="cpu"), _OldPool(9, 3)
    torch.manual_seed(1)
    got = list(starmap(new, batches))
    torch.manual_seed(1)
    ref = list(starmap(old, batches))
    assert new.lq.dtype == torch.uint8
    assert new.gt.dtype == (torch.uint8 if gt_exact else torch.float32)
    for (a, b), (c, d) in zip(got, ref, strict=True):
        assert a.dtype == b.dtype == torch.float32
        assert torch.equal(a, c)
        assert torch.equal(b, d)


# D-17: thread hygiene in workers
class _Threads(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, i):
        return cv2.getNumThreads(), torch.get_num_threads()


def test_worker_threads():
    cv2_threads, torch_threads = next(iter(DataLoader(_Threads(), batch_size=2, num_workers=1,
                                                      worker_init_fn=worker_init_fn)))  # fmt: skip
    assert cv2_threads.tolist() == [1, 1]  # setNumThreads(0): sequential, reported as 1
    assert torch_threads.tolist() == [1, 1]
