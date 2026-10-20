import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from neosr.data import (
    build_dataloader,
    build_dataset,
    degradations,
    seed_data_rngs,
    worker_init_fn,
)
from neosr.data.augmentations import apply_augment, cutblur, cutmix, mixup, resizemix
from neosr.data.data_sampler import EnlargedSampler
from neosr.data.data_util import read_img
from neosr.data.degradations import generate_gaussian_noise_pt
from neosr.data.file_client import FileClient
from neosr.data.otf_dataset import otf_crop_size
from neosr.data.prefetch_dataloader import CUDAPrefetcher
from neosr.data.transforms import paired_random_crop


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


@pytest.mark.parametrize(("scale", "patch_size", "crop"), [(3, 64, 513), (3, 200, 600)])
def test_otf_item(data_root, scale, patch_size, crop):
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
    })
    assert ds.crop_pad_size == crop
    item = ds[0]
    assert Path(item["gt_path"]) == data_root / "GT" / "img_00.png"
    assert item["gt"].shape == (3, crop, crop)  # 256px GT padded up to the crop size
    for k in ("kernel1", "kernel2", "sinc_kernel"):
        assert item[k].shape == (21, 21)
        assert item[k].device.type == "cpu"
        assert item[k].dtype == torch.float32


def test_read_img_errors(tmp_path):
    client = FileClient("disk")
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="bad.png"):
        read_img(client, str(bad), "gt")
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="empty.png"):
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
    seed_data_rngs(0)
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

    def draw(epoch, rank=0):
        sampler = EnlargedSampler(ds, num_replicas=2, rank=rank, ratio=2)
        sampler.set_epoch(epoch)
        return list(sampler)

    assert draw(3) == draw(3)
    assert draw(3) != draw(4)
    both = sorted(draw(5, 0) + draw(5, 1))
    assert both == sorted(ds * 2)


def test_otf_crop_size():
    assert otf_crop_size(64, 4) == 512
    assert otf_crop_size(256, 4) == 1024
    assert otf_crop_size(64, 3) == 513
    assert otf_crop_size(200, 3) == 600
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


class _RngDraws(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return degradations.rng.random()


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


@pytest.mark.parametrize(
    ("dist", "num_gpu", "workers", "batch", "nworkers"),
    [
        (False, 1, 2, 4, 2),
        (False, 2, 1, 8, 2),  # DataParallel: one batch over all GPUs
        (True, 2, 1, 4, 1),  # DDP: per-process sizes
        (False, 1, 0, 4, 0),
    ],
)
def test_build_dataloader(dist, num_gpu, workers, batch, nworkers):
    ds = _Synthetic()
    opt = {"phase": "train", "batch_size": 4, "num_worker_per_gpu": workers}
    # DDP rank 1 of 2: the sampler needs no process group
    sampler = EnlargedSampler(ds, num_replicas=2, rank=1) if dist else None
    loader = build_dataloader(ds, opt, num_gpu=num_gpu, dist=dist, sampler=sampler)
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
    loader = build_dataloader(ds, opt, num_gpu=2, dist=True)
    assert loader.num_workers == max((os.cpu_count() or 1) // 2, 1)
    assert not loader.pin_memory
    assert not loader.persistent_workers
    val = build_dataloader(ds, {"phase": "val"})
    assert (val.batch_size, val.num_workers, val.pin_memory) == (1, 0, False)
    with pytest.raises(ValueError, match="phase"):
        build_dataloader(ds, {"phase": "eval"})


def test_loader_persistent_workers(paired_opt):
    ds = build_dataset(paired_opt)
    sampler = EnlargedSampler(ds, num_replicas=1, rank=0)
    loader = build_dataloader(ds, paired_opt, num_gpu=1, dist=False, sampler=sampler)
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
    prefetcher = CUDAPrefetcher(loader, {})  # type: ignore[arg-type]
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
