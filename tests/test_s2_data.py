"""Stage 2, data lane (docs/refactor/stage2/analysis/data.md): D-01 .. D-20."""

import logging
import os
import pickle
import random
import subprocess
import sys
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
from neosr.data.augmentations import apply_augment, resizemix
from neosr.data.data_util import to_float32
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


# tight canvas 128 (crop) and 416 (pad), legacy 512 (pad) of a 200x296 GT
@pytest.mark.parametrize(("patch", "tight"), [(24, True), (96, True), (24, False)])
def test_otf_crop_before_float_is_exact(pairs, monkeypatch, patch, tight):
    new = _items(build_dataset(_otf_opt(pairs, patch, tight_canvas=tight)))
    _float_first(monkeypatch, otf_dataset)
    old = _items(build_dataset(_otf_opt(pairs, patch, tight_canvas=tight)))
    _assert_items_equal(new, old)
    assert new[0]["gt"].shape == (3, *(otf_dataset.otf_crop_size(patch, 4, tight=tight),) * 2)


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


# D-02: canvas key, opt-in until an A/B
def test_otf_tight_canvas_key(pairs):
    assert build_dataset(_otf_opt(pairs, 64)).crop_pad_size == 512
    assert build_dataset(_otf_opt(pairs, 64, tight_canvas=True)).crop_pad_size == 288


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


# D-04: resizemix targets without aliasing (opt-in)
@pytest.mark.parametrize("antialias", [True, False])
def test_resizemix_antialias(antialias):
    """Downscaled white noise: antialiasing averages it out (std 0.06 at 4x), plain bicubic
    samples it (aliasing, std 0.21). The default keeps the old plain bicubic."""
    gt = torch.rand(2, 3, 64, 64, generator=torch.Generator().manual_seed(0))
    kw = {"antialias": True} if antialias else {}
    seed_rng(0)
    out, _ = resizemix(gt.clone(), gt[..., ::2, ::2].clone(), scope=(0.25, 0.25), scale=2, **kw)
    box = (out != gt).all(1)
    assert box.sum(dim=(1, 2)).min() >= 8 * 8  # the 16x16 box, at most halved by the border
    assert (out[:, 0][box].std() < 0.1) == antialias


def test_apply_augment_passes_antialias():
    gt = torch.rand(2, 3, 32, 32, generator=torch.Generator().manual_seed(0))
    lq = gt[..., ::2, ::2].clone()
    outs = []
    for aa in (False, True):
        seed_rng(0)
        outs.append(apply_augment(gt, lq, scale=2, augs=("resizemix",), prob=(1,), antialias=aa))
    box = (outs[0][0] != gt).all(1)
    assert box.any()
    assert torch.equal(box, (outs[1][0] != gt).all(1))  # same box, different filter
    assert not torch.equal(outs[0][0], outs[1][0])


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

    def __init__(self, size, batch, device="cpu"):
        self.batch, self.size, self.ptr, self.device = batch, size // batch * batch, 0, device
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
        idx = torch.randperm(self.size, device=self.device)[:b]
        out_lq, out_gt = self.lq[idx], self.gt[idx]
        self.lq[idx], self.gt[idx] = lq, gt
        return out_lq, out_gt


DEVICES = ["cpu", *(["cuda"] if CUDA else [])]


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("gt_8bit", [True, False])
def test_pair_pool_uint8_bit_exact(dev, gt_8bit):
    """The producers as in training: lq quantised by the Degrader on the device, gt decoded by the
    dataset (numpy to_float32) and copied over; luma gt is not 8 bit."""
    gen = np.random.default_rng(0)

    def batch():
        lq = torch.from_numpy(gen.random((3, 3, 4, 4), np.float32)).to(dev)
        lq = torch.clamp((lq * 255.0).round(), 0, 255) / 255.0  # Degrader._quantize
        if gt_8bit:
            gt = torch.from_numpy(to_float32(gen.integers(0, 256, (3, 3, 16, 16), np.uint8)))
        else:
            gt = torch.from_numpy(gen.random((3, 1, 16, 16), np.float32))
        return lq, gt.to(dev)

    batches = [batch() for _ in range(12)]
    new, old = PairPool(9, 3, device=dev, gt_8bit=gt_8bit), _OldPool(9, 3, dev)
    torch.manual_seed(1)
    got = list(starmap(new, batches))
    torch.manual_seed(1)
    ref = list(starmap(old, batches))
    assert new.lq.dtype == torch.uint8
    assert new.gt.dtype == (torch.uint8 if gt_8bit else torch.float32)
    for (a, b), (c, d) in zip(got, ref, strict=True):
        assert a.dtype == b.dtype == torch.float32
        assert torch.equal(a, c)
        assert torch.equal(b, d)


def test_pair_pool_storage_not_from_values():
    """A first gt batch that happens to be 8 bit (black luma) must not switch the pool to uint8:
    later luma values such as 0.299 would come back rounded (0.298039)."""
    pool = PairPool(2, 2, device="cpu")
    lq, black, luma = torch.zeros(2, 1, 2, 2), torch.zeros(2, 1, 8, 8), torch.full((2, 1, 8, 8), 0.299)
    pool(lq, black)  # fills the pool
    assert torch.equal(pool(lq, luma)[1], black)  # swaps luma in
    assert pool.gt.dtype == torch.float32
    assert torch.equal(pool(lq, black)[1], luma)


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


# D-07 / D-08: lmdb keys, encoding, streaming atomic build
def _lmdb_src(root: Path, names: list[str]) -> None:
    root.mkdir()
    for i, n in enumerate(names):
        cv2.imwrite(str(root / n), np.full((6, 5, 3), 10 * i, np.uint8))


@pytest.mark.parametrize("pool", [False, True])
def test_lmdb_round_trip(tmp_path, pool):
    import create_lmdb  # noqa: PLC0415
    from neosr.utils.lmdb_util import make_lmdb_from_imgs  # noqa: PLC0415

    names = ["a.b.png", "a.png", "my photo.png", "ü.png", "X.PNG", "z.jpg"]
    _lmdb_src(tmp_path / "src", [*names, ".hidden.png"])
    (tmp_path / "src" / "notes.txt").write_text("x")
    paths, keys = create_lmdb.prepare_keys(str(tmp_path / "src"))
    assert keys == ["X", "a.b", "a", "my photo", "z", "ü"]
    out = str(tmp_path / "db.lmdb")
    # map_size far too small: the build grows it
    make_lmdb_from_imgs(str(tmp_path / "src"), out, paths, keys, batch=2,
                        multiprocessing_read=pool, n_thread=2, map_size=1 << 12)  # fmt: skip
    assert du.lmdb_keys(out) == keys
    ds = build_dataset({"type": "single", "phase": "test", "dataroot_lq": out})
    vals = [round(float(ds[i]["lq"].mean()) * 255) for i in range(len(ds))]
    assert vals == [10 * names.index(p) for p in paths]
    assert not Path(f"{out}.tmp").exists()


@pytest.mark.parametrize("pool", [False, True])
def test_lmdb_corrupt_leaves_nothing(tmp_path, pool):
    from neosr.utils.lmdb_util import make_lmdb_from_imgs  # noqa: PLC0415

    names = [f"{i}.png" for i in range(6)]
    _lmdb_src(tmp_path / "src", names)
    (tmp_path / "src" / "3.png").write_bytes(b"broken")
    out = str(tmp_path / "db.lmdb")
    with pytest.raises(OSError, match=r"3\.png"):
        make_lmdb_from_imgs(str(tmp_path / "src"), out, names, [n[0] for n in names],
                            batch=2, multiprocessing_read=pool, n_thread=2)  # fmt: skip
    assert sorted(p.name for p in tmp_path.iterdir()) == ["src"]


def test_lmdb_duplicate_keys_rejected(tmp_path):
    import create_lmdb  # noqa: PLC0415
    from neosr.utils.lmdb_util import make_lmdb_from_imgs  # noqa: PLC0415

    _lmdb_src(tmp_path / "src", ["a.png", "a.jpg"])
    paths, keys = create_lmdb.prepare_keys(str(tmp_path / "src"))
    with pytest.raises(ValueError, match="Duplicate"):
        make_lmdb_from_imgs(str(tmp_path / "src"), str(tmp_path / "db.lmdb"), paths, keys)
    assert not (tmp_path / "db.lmdb").exists()


# D-06 (b): tile tool
def test_extract_subimages(tmp_path):
    from scripts import extract_subimages as ex  # noqa: PLC0415

    assert ex.starts(10, 4, 3) == [0, 3, 6]
    assert ex.starts(11, 4, 3) == [0, 3, 6, 7]
    assert ex.starts(3, 4, 3) == [0]
    img = _img(11, 10, 0)
    cv2.imwrite(str(tmp_path / "im.png"), img)
    (tmp_path / "out").mkdir()
    assert ex.tile(str(tmp_path / "im.png"), tmp_path / "out", 4, 3, 1) == 12
    t = cv2.imread(str(tmp_path / "out" / "im_s012.png"))
    assert np.array_equal(t, img[7:11, 6:10])
    with pytest.raises(OSError, match=r"x\.png"):
        ex.tile(str(tmp_path / "x.png"), tmp_path / "out", 4, 3, 1)


# D-14: 16-bit sources stored as the 8 bit the loaders decode: bit-identical training input
def test_extract_subimages_refuses_shared_stems(tmp_path):
    """a.png and a.jpg would both write a_s001.png, ... (racing across workers)."""
    (tmp_path / "src").mkdir()
    for n in ("a.png", "a.jpg", "b.png"):
        cv2.imwrite(str(tmp_path / "src" / n), _img(8, 8, 0))
    run = subprocess.run([sys.executable, "-m", "scripts.extract_subimages", "-i", str(tmp_path / "src"),
                          "-o", str(tmp_path / "out"), "--crop", "4", "--step", "4", "--workers", "1"],
                         cwd=Path(__file__).parents[1], capture_output=True, text=True, check=False)  # fmt: skip
    assert run.returncode != 0
    assert "share a file stem" in run.stderr
    assert not (tmp_path / "out").exists()


def test_extract_subimages_bits8(pairs, tmp_path):
    from scripts import extract_subimages as ex  # noqa: PLC0415

    src = pairs / "GT" / "p1.png"  # 16-bit
    assert cv2.imread(str(src), cv2.IMREAD_UNCHANGED).dtype == np.uint16
    out = tmp_path / "out8"
    out.mkdir()
    assert ex.tile(str(src), out, 0, 0, 1, bits8=True) == 1
    stored = cv2.imread(str(out / "p1.png"), cv2.IMREAD_UNCHANGED)
    assert stored.dtype == np.uint8
    assert np.array_equal(du.imfrombytes(src.read_bytes()), du.imfrombytes((out / "p1.png").read_bytes()))


# D-20: no cudnn autotuning per image size inside the Degrader
def test_degrader_disables_cudnn_benchmark(monkeypatch):
    from neosr.data import otf_degradation as od  # noqa: PLC0415

    seen, conv = [], od.filter2D
    monkeypatch.setattr(od, "filter2D", lambda *a: seen.append(torch.backends.cudnn.benchmark) or conv(*a))
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    d = Degrader({**{k: v for k, v in DEG.items() if not k.startswith("jpeg")}, "jpeg_compress": False},
                 scale=4, device="cpu")  # fmt: skip
    ks = [torch.full((1, n, n), 1 / n**2) for n in (7, 9, 21)]
    _seed(0)
    d(torch.rand(1, 3, 96, 96), *ks)
    assert seen
    assert not any(seen)
    assert torch.backends.cudnn.benchmark
    with pytest.raises(ValueError, match="kernel size"):
        d(torch.rand(1, 3, 96, 96), *ks[:2], torch.ones(1, 4, 4))  # restored on errors too
    assert torch.backends.cudnn.benchmark


# D-12 / F13: dataset normalisation is refused, not silently applied or ignored
@pytest.mark.parametrize("kind", ["paired", "single", "otf"])
@pytest.mark.parametrize("norm", [{"mean": [0.5] * 3, "std": [0.5] * 3}, {"mean": [0.5] * 3}, {"std": [1.0] * 3}])
def test_normalization_rejected(pairs, kind, norm):
    opt = {"paired": lambda: _paired_opt(pairs, "train"), "otf": lambda: _otf_opt(pairs),
           "single": lambda: {"type": "single", "phase": "test", "dataroot_lq": str(pairs / "LQ")}}[kind]()  # fmt: skip
    with pytest.raises(ConfigError, match=next(iter(norm))):
        build_dataset({**opt, **norm})


@pytest.mark.parametrize("color", [None, "y"])
def test_otf_feed_path(pairs, color):
    """The OTF training feed as models/otf.py must run it (trainer handoff): dataset RGB items ->
    Degrader.degrade -> crop -> pool -> a generator with the configured channel count."""
    from torch.utils.data import default_collate  # noqa: PLC0415

    from neosr.data.transforms import paired_random_crop  # noqa: PLC0415

    ds = build_dataset(_otf_opt(pairs, 24, tight_canvas=True, color=color))
    d = Degrader({**DEG, "color": color}, scale=4, device="cpu")
    pool = PairPool(2, 2, device="cpu", gt_8bit=not d.gray)
    c = 1 if color else 3
    net = torch.nn.Sequential(torch.nn.Conv2d(c, 16 * c, 3, padding=1), torch.nn.PixelShuffle(4))
    _seed(0)
    for _ in range(2):  # fill, then swap
        data = default_collate([ds[0], ds[1]])
        gt, lq = d.degrade(data["gt"], data["kernel1"], data["kernel2"], data["sinc_kernel"])
        gt, lq = paired_random_crop([gt], lq, 24, 4)
        lq, gt = pool(lq, gt)
        assert lq.shape == (2, c, 24, 24)
        assert net(lq).shape == gt.shape == (2, c, 96, 96)
    assert pool.gt.dtype == (torch.float32 if color else torch.uint8)
