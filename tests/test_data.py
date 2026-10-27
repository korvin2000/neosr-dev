import os
import random
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from neosr.config import ConfigError
from neosr.data import build_dataloader, build_dataset, degradations, worker_init_fn
from neosr.data.augmentations import apply_augment, cutblur, cutmix, mixup, resizemix
from neosr.data.data_sampler import EnlargedSampler
from neosr.data.data_util import (
    paired_paths_from_folder,
    paired_paths_from_lmdb,
    read_img,
)
from neosr.data.degradations import generate_gaussian_noise_pt
from neosr.data.file_client import FileClient
from neosr.data.otf_dataset import otf_crop_size
from neosr.data.prefetch_dataloader import CUDAPrefetcher
from neosr.data.transforms import paired_random_crop
from neosr.runtime import Runtime
from neosr.utils.rng import get_state, rng, seed_rng, set_state


def write_pairs(
    root: Path, n: int = 8, gt_size: int = 256, scale: int = 4
) -> list[str]:
    rng = np.random.default_rng(0)
    (root / "GT").mkdir(parents=True)
    (root / "LQ").mkdir()
    names = [f"img_{i:02d}.png" for i in range(n)]
    for name in names:
        gt = rng.integers(0, 256, (gt_size, gt_size, 3), dtype=np.uint8)
        cv2.imwrite(str(root / "GT" / name), gt)
        lq = cv2.resize(gt, (gt_size // scale,) * 2, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(root / "LQ" / name), lq)
    return names


@pytest.fixture(scope="module")
def data_root(tmp_path_factory) -> Path:
    # a root with spaces: paths must be joined, never split on whitespace
    root = tmp_path_factory.mktemp("data") / "data root"
    names = write_pairs(root)
    (root / "meta_info.txt").write_text("".join(f"{n} (256,256,3)\n" for n in names))
    return root


@pytest.fixture
def paired_opt(data_root):
    return {
        "type": "paired",
        "phase": "train",
        "dataroot_gt": str(data_root / "GT"),
        "dataroot_lq": str(data_root / "LQ"),
        "meta_info": str(data_root / "meta_info.txt"),
        "scale": 4,
        "patch_size": 64,
        "batch_size": 4,
        "num_worker_per_gpu": 2,
    }


def upsample_nearest(lq: torch.Tensor, scale: int) -> torch.Tensor:
    return lq.repeat_interleave(scale, -2).repeat_interleave(scale, -1)


@pytest.mark.parametrize("key", ["meta_info", "meta_info_file"])
def test_paired_item_shapes(paired_opt, data_root, key):
    opt = dict(paired_opt)
    opt[key] = opt.pop("meta_info")
    ds = build_dataset(opt)
    assert len(ds) == 8
    item = ds[0]
    assert item["gt"].shape == (3, 256, 256)
    assert item["lq"].shape == (3, 64, 64)
    # meta_info mode used to drop the file name from the LQ path
    assert Path(item["lq_path"]).parent == data_root / "LQ"
    assert Path(item["lq_path"]).name == Path(item["gt_path"]).name


@pytest.mark.parametrize("key", ["meta_info", "meta_info_file"])
def test_single_meta_info(data_root, key):
    ds = build_dataset({
        "type": "single",
        "phase": "test",
        "dataroot_lq": str(data_root / "LQ"),
        key: str(data_root / "meta_info.txt"),
    })
    assert len(ds) == 8
    item = ds[0]
    assert item["lq"].shape == (3, 64, 64)
    assert Path(item["lq_path"]) == data_root / "LQ" / "img_00.png"


def test_lmdb_dotted_keys(tmp_path):
    """R3.2: keys keep their inner dots; 'scene.001' and 'scene.002' stay two records."""
    from neosr.utils.lmdb_util import make_lmdb_from_imgs  # noqa: PLC0415

    names, keys = ["a.png", "b.png"], ["scene.001", "scene.002"]
    for i, name in enumerate(names):
        cv2.imwrite(str(tmp_path / name), np.full((8, 8, 3), 60 * (i + 1), np.uint8))
    for sub in ("lq.lmdb", "gt.lmdb"):
        make_lmdb_from_imgs(str(tmp_path), str(tmp_path / sub), names, keys,
                            multiprocessing_read=False, map_size=1 << 20)  # fmt: skip
    pairs = paired_paths_from_lmdb(
        [str(tmp_path / "lq.lmdb"), str(tmp_path / "gt.lmdb")], ["lq", "gt"]
    )
    assert [p["gt_path"] for p in pairs] == keys
    ds = build_dataset({
        "type": "single",
        "phase": "test",
        "dataroot_lq": str(tmp_path / "lq.lmdb"),
    })
    assert list(ds.paths) == keys  # a PathList (D-18)
    assert [round(float(ds[i]["lq"].mean()) * 255) for i in range(2)] == [60, 120]


@pytest.mark.parametrize("rel", [False, True])
def test_folder_pairing_repeated_root_name(tmp_path, monkeypatch, rel):
    """R3.4: only the root is swapped, not every occurrence of its name ('GT/image_GT.png')."""
    for sub in ("GT", "LQ"):
        (tmp_path / sub / "x_GT").mkdir(parents=True)
        for f in ("image_GT.png", "x_GT/c.png"):
            cv2.imwrite(str(tmp_path / sub / f), np.zeros((4, 4, 3), np.uint8))
    monkeypatch.chdir(tmp_path)
    root = Path("./") if rel else tmp_path
    pairs = paired_paths_from_folder(
        [str(root / "LQ"), f"{root / 'GT'}/"], ["lq", "gt"]
    )
    got = sorted(
        (Path(p["lq_path"]).relative_to(root / "LQ").as_posix(),
         Path(p["gt_path"]).relative_to(root / "GT").as_posix()) for p in pairs
    )  # fmt: skip
    assert got == [("image_GT.png",) * 2, ("x_GT/c.png",) * 2]


@pytest.mark.parametrize(
    ("scale", "patch_size", "legacy", "crop"),
    [(3, 64, False, 225), (3, 200, False, 633), (3, 64, True, 513), (3, 200, True, 600)],
)
def test_otf_item(data_root, scale, patch_size, legacy, crop):
    kernels = {
        "kernel_list": ["iso", "aniso"],
        "kernel_prob": [0.5, 0.5],
        "blur_sigma": [0.2, 3],
        "betag_range": [0.5, 4],
        "betap_range": [1, 2],
    }
    ds = build_dataset({
        "type": "otf",
        "phase": "train",
        "dataroot_gt": str(data_root / "GT"),
        "meta_info": str(data_root / "meta_info.txt"),
        "scale": scale,
        "patch_size": patch_size,
        **kernels,
        **{f"{k}2": v for k, v in kernels.items()},
        "sinc_prob": 0.5,
        "sinc_prob2": 0.5,
        "final_sinc_prob": 0.5,
        "legacy_canvas": legacy,
    })
    assert ds.crop_pad_size == crop
    item = ds[0]
    assert Path(item["gt_path"]) == data_root / "GT" / "img_00.png"
    assert item["gt"].shape == (3, crop, crop)  # 256px GT cropped or padded to the canvas
    for k in ("kernel1", "kernel2", "sinc_kernel"):
        assert item[k].shape == (21, 21)
        assert item[k].device.type == "cpu"
        assert item[k].dtype == torch.float32


def test_read_img_errors(tmp_path):
    client = FileClient("disk")
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(OSError, match="bad.png"):
        read_img(client, str(bad), "gt")
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    with pytest.raises(OSError, match="empty.png"):
        read_img(client, str(empty), "gt")
    with pytest.raises(FileNotFoundError):
        read_img(client, str(tmp_path / "missing.png"), "gt")


def test_paired_random_crop_raises():
    gt, lq = np.zeros((30, 30, 3)), np.zeros((8, 8, 3))
    with pytest.raises(ValueError, match="Scale mismatches"):
        paired_random_crop(gt, lq, 4, 4, "x.png")
    with pytest.raises(ValueError, match="smaller than patch size"):
        paired_random_crop(np.zeros((32, 32, 3)), lq, 16, 4, "x.png")


def test_augment_none_is_identity():
    gt, lq = torch.rand(4, 3, 32, 32), torch.rand(4, 3, 8, 8)
    out_gt, out_lq = apply_augment(gt, lq, scale=4, augs=("none",), prob=(1.0,))
    assert out_gt is gt
    assert out_lq is lq
    # zero weights are never drawn, also not in the multi-augment branch
    for _ in range(20):
        out_gt, out_lq = apply_augment(
            gt, lq, scale=4, augs=("none", "mixup"), prob=(1.0, 0.0), multi_prob=1.0
        )
        assert out_gt is gt
        assert out_lq is lq


def test_augment_weighted_choice():
    seed_rng(0)
    gt, lq = (
        torch.rand(2, 3, 16, 16, dtype=torch.float16),
        torch.rand(2, 3, 8, 8, dtype=torch.float16),
    )
    unchanged = 0
    for _ in range(400):
        out_gt, out_lq = apply_augment(
            gt, lq, scale=2, augs=("none", "cutmix"), prob=(1.0, 3.0), multi_prob=0.0
        )
        assert (out_gt.dtype, out_lq.dtype) == (torch.float16, torch.float16)
        unchanged += out_gt is gt
    assert 70 < unchanged < 130  # 'none' has weight 1/4


def test_mixup_mixes_lq_with_lq():
    b = 4
    lq = torch.arange(b, dtype=torch.float32).view(b, 1, 1, 1).expand(b, 3, 8, 8)
    gt = upsample_nearest(lq, 4) + 10  # GT values never appear in LQ
    out_gt, out_lq = mixup(gt, lq.clone(), scale=4)
    assert out_lq.max() <= b - 1
    # the same partner and ratio for GT and LQ
    torch.testing.assert_close(out_gt - 10, upsample_nearest(out_lq, 4))


@pytest.mark.parametrize("scale", [1, 2, 4])
@pytest.mark.parametrize("hw", [(8, 12), (1, 3)])
def test_box_augments_keep_alignment(scale, hw):
    lq = torch.rand(4, 3, *hw)
    gt = upsample_nearest(lq, scale)
    for aug in (cutmix, mixup):
        out_gt, out_lq = aug(gt.clone(), lq.clone(), scale=scale)
        torch.testing.assert_close(out_gt, upsample_nearest(out_lq, scale))
    for aug in (resizemix, cutblur):
        out_gt, out_lq = aug(gt.clone(), lq.clone(), scale=scale)
        assert out_gt.shape == gt.shape
        assert out_lq.shape == lq.shape
        assert 0 <= out_lq.min() <= out_lq.max() <= 1
    gt_in, lq_in = gt.clone(), lq.clone()
    apply_augment(gt, lq, scale=scale, prob=None)
    assert torch.equal(gt, gt_in)
    assert torch.equal(lq, lq_in)


def test_augment_validation():
    gt, lq = torch.rand(2, 3, 8, 8), torch.rand(2, 3, 4, 4)
    with pytest.raises(ValueError, match="don't match"):
        apply_augment(gt, lq, scale=2, augs=("none", "mixup"), prob=(1.0,))
    with pytest.raises(ValueError, match="Unknown"):
        apply_augment(gt, lq, scale=2, augs=("cutout",), prob=(1.0,))
    with pytest.raises(ValueError, match="must be"):
        mixup(gt, lq, scale=4)


def test_sampler_deterministic_per_epoch():
    ds = list(range(10))

    def draw(epoch, rank=0, replicas=2, seed=7):
        sampler = EnlargedSampler(ds, replicas, rank, 2, seed)
        sampler.set_epoch(epoch)
        return list(sampler)

    assert draw(3) == draw(3)
    assert draw(3) != draw(4)
    assert draw(3) != draw(3, seed=8)
    # both ranks shard one permutation
    full = draw(5, replicas=1)
    assert (draw(5, 0), draw(5, 1)) == (full[0::2], full[1::2])
    assert sorted(full) == sorted(ds * 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_sampler_no_cuda_allocation():
    sampler = EnlargedSampler(list(range(10)), seed=3)
    expected = list(sampler)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    with torch.device("cuda"):  # the legacy trainer still sets a CUDA default device
        assert list(sampler) == expected
    assert torch.cuda.max_memory_allocated() == before


def test_otf_crop_size():
    # D-02: GT patch + 32, rounded up to a multiple of scale
    assert otf_crop_size(64, 4) == 288
    assert otf_crop_size(256, 4) == 1056
    assert otf_crop_size(64, 3) == 225
    assert otf_crop_size(200, 3) == 633
    # legacy_canvas: the old floor of 512
    assert otf_crop_size(64, 4, legacy=True) == 512
    assert otf_crop_size(256, 4, legacy=True) == 1024
    assert otf_crop_size(64, 3, legacy=True) == 513
    assert otf_crop_size(200, 3, legacy=True) == 600
    with pytest.raises(ValueError, match="patch_size"):
        otf_crop_size(None, 4)  # type: ignore[arg-type]


def test_gray_noise_per_sample():
    img = torch.zeros(4, 3, 16, 16)
    noise = generate_gaussian_noise_pt(
        img, sigma=torch.full((4,), 10.0), gray_noise=torch.ones(4)
    )
    assert noise.shape == img.shape
    # gray: equal over channels, but independent between samples
    torch.testing.assert_close(noise[:, 0], noise[:, 1])
    assert not torch.allclose(noise[0], noise[1])


@pytest.mark.parametrize("c", [1, 3])
@pytest.mark.parametrize("gray", [1, torch.ones(4), torch.tensor([1.0, 0.0, 1.0, 0.0])])
@pytest.mark.parametrize(
    "gen",
    [degradations.generate_gaussian_noise_pt, degradations.generate_poisson_noise_pt],
)
def test_gray_noise_matches_channels(gen, gray, c):
    """R3.3: gray noise follows the input's channel count (color='y' batches are 1-channel)."""
    img = torch.rand(4, c, 16, 16, generator=torch.Generator().manual_seed(0))
    noise = gen(img, torch.full((4,), 0.5), gray)
    assert noise.shape == img.shape
    rows = [0, 1, 2, 3] if isinstance(gray, int) else gray.nonzero().flatten().tolist()
    torch.testing.assert_close(noise[rows, :1].expand(-1, c, -1, -1), noise[rows])


class _RngDraws(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return rng.random()


def test_worker_rngs_distinct():
    loader = DataLoader(
        _RngDraws(), batch_size=1, num_workers=2, worker_init_fn=worker_init_fn
    )
    draws = [float(x) for x in loader]
    # worker 0 serves items 0 and 2, worker 1 items 1 and 3
    assert draws[0] != draws[1]
    assert len(set(draws)) == 4


class _Synthetic(Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, index):
        return {"gt": torch.full((3, 4, 4), float(index)), "idx": index}


def _rt(rank: int = 0, world_size: int = 1, seed: int = 0) -> Runtime:
    return Runtime(
        torch.device("cpu"), rank, world_size, seed, distributed=world_size > 1
    )


@pytest.mark.parametrize(
    ("dist", "workers", "batch", "nworkers"),
    [
        (False, 2, 4, 2),
        (True, 1, 4, 1),  # DDP: per-process sizes
        (False, 0, 4, 0),
    ],
)
def test_build_dataloader(dist, workers, batch, nworkers):
    ds = _Synthetic()
    opt = {"phase": "train", "batch_size": 4, "num_worker_per_gpu": workers}
    # DDP rank 1 of 2: the sampler needs no process group
    sampler = EnlargedSampler(ds, num_replicas=2, rank=1) if dist else None
    rt = _rt(1, 2) if dist else _rt()
    loader = build_dataloader(ds, opt, runtime=rt, sampler=sampler)
    assert loader.batch_size == batch
    assert loader.num_workers == nworkers
    assert loader.drop_last
    if dist:
        assert loader.sampler is sampler
    else:
        assert isinstance(loader.sampler, RandomSampler)
    assert loader.persistent_workers == (nworkers > 0)
    assert loader.prefetch_factor == (2 if nworkers else None)
    assert loader.pin_memory == torch.cuda.is_available()
    seen = [int(i) for b in loader for i in b["idx"]]
    assert len(seen) == (len(ds) // 2 if dist else len(ds)) // batch * batch
    if dist:
        assert set(seen) <= set(sampler)  # type: ignore[arg-type]


def test_build_dataloader_overrides():
    ds = _Synthetic()
    opt = {
        "phase": "train",
        "batch_size": 2,
        "num_worker_per_gpu": "auto",
        "pin_memory": False,
        "persistent_workers": False,
    }
    loader = build_dataloader(ds, opt, runtime=_rt(0, 2))
    assert loader.num_workers == max((os.process_cpu_count() or 1) // 2 - 1, 1)  # D-17
    assert not loader.pin_memory
    assert not loader.persistent_workers
    val = build_dataloader(ds, {"phase": "val"}, runtime=_rt())
    assert (val.batch_size, val.num_workers, val.drop_last) == (1, 0, False)
    assert val.pin_memory == torch.cuda.is_available()
    with pytest.raises(ValueError, match="phase"):
        build_dataloader(ds, {"phase": "eval"}, runtime=_rt())


def test_loader_persistent_workers(paired_opt):
    ds = build_dataset(paired_opt)
    sampler = EnlargedSampler(ds, num_replicas=1, rank=0)
    loader = build_dataloader(ds, paired_opt, runtime=_rt(), sampler=sampler)
    assert loader.num_workers == 2
    assert loader.persistent_workers
    iterators = []
    for epoch in range(2):
        sampler.set_epoch(epoch)
        it = iter(loader)
        batches = list(it)
        iterators.append(it)
        assert len(batches) == 2
        assert batches[0]["gt"].shape == (4, 3, 256, 256)
        assert batches[0]["lq"].shape == (4, 3, 64, 64)
    assert iterators[0] is iterators[1]  # workers were reused, not respawned


class _FreshBatches:
    """Yields new dicts on every pass, like a DataLoader."""

    def __init__(self, n: int, numel: int) -> None:
        # pinned like a pin_memory loader, and allocated up front: cudaHostAlloc
        # synchronizes the device and would hide the race
        self.src = [torch.full((numel,), float(i)).pin_memory() for i in range(n)]
        self.produced = 0

    def __iter__(self):
        for i, x in enumerate(self.src):
            self.produced += 1
            yield {"x": x, "p": str(i)}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_prefetcher():

    n, numel = 6, 1 << 18
    loader = _FreshBatches(n, numel)
    prefetcher = CUDAPrefetcher(loader, torch.device("cuda"))  # type: ignore[arg-type]
    assert loader.produced == 0  # construction must not consume a batch
    for epoch in range(2):  # reset starts each epoch
        prefetcher.reset()
        sums = []
        while (batch := prefetcher.next()) is not None:
            assert batch["x"].is_cuda
            assert batch["p"] == str(len(sums))
            # delay the consumer read past the next preload, then drop the batch:
            # without record_stream the allocator hands its memory to that preload,
            # whose copy overwrites it before the sum runs
            torch.cuda._sleep(20_000_000)
            sums.append(batch["x"].sum())
            del batch
        torch.cuda.synchronize()
        assert [s.item() for s in sums] == [float(i * numel) for i in range(n)]
        assert loader.produced == n * (epoch + 1)


def test_rng_reseeded_in_place():
    bound = degradations.rng  # a consumer binding made at import time
    seed_rng(5)
    first = bound.random(3)
    seed_rng(5)
    assert (rng.random(3) == first).all()
    state = get_state()
    later = rng.random(2)
    set_state(state)
    assert (bound.random(2) == later).all()
    assert bound is rng


class _Draws(Dataset):
    """Every stream a dataset draws from: shuffling order, rng, random and torch."""

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return {
            "idx": index,
            "np": rng.random(),
            "py": random.random(),
            "t": torch.rand(()),
        }


def _loader(workers: int, seed: int = 11, **cfg) -> DataLoader:
    runtime = Runtime(torch.device("cpu"), 0, 1, seed, distributed=False)
    opt = {"phase": "train", "batch_size": 4, "num_worker_per_gpu": workers, **cfg}
    return build_dataloader(_Draws(), opt, runtime=runtime)


def _flat(batch) -> list[float]:
    return [float(x) for k in ("idx", "np", "py", "t") for x in batch[k]]


def test_build_dataloader_runtime():
    ds = _Synthetic()
    opt = {"phase": "train", "batch_size": 2, "num_worker_per_gpu": "auto"}
    rt = Runtime(torch.device("cpu"), 1, 2, 100, distributed=True)
    loader = build_dataloader(ds, opt, runtime=rt)
    assert (loader.batch_size, loader.num_workers) == (
        2,
        max((os.process_cpu_count() or 1) // 2 - 1, 1),  # D-17
    )
    assert loader.generator.initial_seed() == 101  # type: ignore[union-attr]
    val = build_dataloader(ds, {"phase": "val"}, runtime=rt)
    assert (val.batch_size, val.num_workers, val.drop_last) == (1, 0, False)


@pytest.mark.parametrize("workers", [0, 2])
def test_identical_settings_identical_first_batch(workers):
    def first(seed=11, torch_seed=0):
        seed_rng(0)  # the main-process streams, which runtime.seed_all seeds
        random.seed(0)
        np.random.seed(0)  # noqa: NPY002
        loader = _loader(workers, seed)
        # global torch must not reach the order or the worker seeds
        torch.manual_seed(torch_seed)
        batch = next(iter(loader))
        return {k: [float(x) for x in batch[k]] for k in ("idx", "np", "py", "t")}

    a = first()
    assert a == first()
    b = first(torch_seed=1)
    owned = ("idx", "np", "py") if workers == 0 else ("idx", "np", "py", "t")
    assert all(a[k] == b[k] for k in owned)
    c = first(seed=12)  # only runtime.seed changes
    assert a["idx"] != c["idx"]
    if workers:  # worker draws derive from the loader generator
        assert not set(a["np"]) & set(c["np"])


@pytest.mark.parametrize(("workers", "persistent"), [(0, False), (2, False), (2, True)])
def test_epochs_differ(workers, persistent):
    seed_rng(0)
    loader = _loader(workers, persistent_workers=persistent)
    assert loader.persistent_workers == persistent
    epochs = [[_flat(b) for b in loader] for _ in range(2)]
    assert len(epochs[0]) == len(epochs[1]) == len(loader) == 2
    # the draws change, not only the order: compare them as sets
    np_draws = [{x for b in e for x in b[4:8]} for e in epochs]
    assert np_draws[0] != np_draws[1]
    assert not np_draws[0] & np_draws[1]


def test_unreadable_image_in_worker(tmp_path, paired_opt):
    """D-01: training skips the bad pair (full batches, a warning with the path); val raises."""
    root = tmp_path / "bad"
    write_pairs(root, n=4)
    (root / "LQ" / "img_02.png").write_bytes(b"not an image")
    opt = {
        **paired_opt,
        "dataroot_gt": str(root / "GT"),
        "dataroot_lq": str(root / "LQ"),
        "num_worker_per_gpu": 1,
    }
    del opt["meta_info"]
    rt = Runtime(torch.device("cpu"), 0, 1, 0, distributed=False)
    loader = build_dataloader(build_dataset(opt), opt, runtime=rt)
    batches = list(loader)
    assert len(batches) == 1
    assert batches[0]["gt"].shape == (4, 3, 256, 256)
    assert not any("img_02" in p for p in batches[0]["lq_path"])
    val = build_dataset({**opt, "phase": "val"})
    with pytest.raises(OSError, match=r"img_02\.png"):
        list(build_dataloader(val, {"phase": "val"}, runtime=rt))


def test_build_dataset_unknown_type():
    with pytest.raises(ConfigError, match=r"datasets\.train.*did you mean .paired."):
        build_dataset({"type": "pairde", "phase": "train"})
    with pytest.raises(ConfigError, match="dataroot_lq"):
        build_dataset({
            "type": "otf",
            "phase": "train",
            "dataroot_gt": "x",
            "dataroot_lq": "y",
        })


@pytest.mark.parametrize("gray", [False, True])
def test_gaussian_noise_numpy(gray):
    img = np.full((8, 6, 3), 0.5, dtype=np.float32)
    seed_rng(1)
    out = degradations.add_gaussian_noise(img, sigma=10, clip=False, gray_noise=gray)
    noise = out - img
    assert out.shape == img.shape
    assert out.dtype == np.float32
    assert noise.std() > 0
    assert np.allclose(noise[..., 0], noise[..., 1]) == gray
    seed_rng(1)
    assert np.array_equal(
        degradations.add_gaussian_noise(img, sigma=10, clip=False, gray_noise=gray), out
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_prefetcher_counts_and_record_stream(monkeypatch):
    calls = []
    record = torch.Tensor.record_stream

    def spy(self, stream):
        calls.append(self.shape)
        return record(self, stream)

    monkeypatch.setattr(torch.Tensor, "record_stream", spy)
    loader = build_dataloader(
        _Synthetic(),
        {"phase": "train", "batch_size": 4, "num_worker_per_gpu": 0},
        runtime=_rt(),
    )
    prefetcher = CUDAPrefetcher(loader, torch.device("cuda", 0))
    for _ in range(2):
        prefetcher.reset()
        n = 0
        while (batch := prefetcher.next()) is not None:
            assert batch["gt"].device == torch.device("cuda", 0)
            n += 1
        assert n == len(loader) == 4
    assert len(calls) == 2 * 4 * 2  # every tensor ("gt" and "idx") of every batch
