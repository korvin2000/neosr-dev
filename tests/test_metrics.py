from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from neosr.metrics import calculate, calculate_metric, release_metric_nets
from neosr.utils.img_util import img2tensor, tensor2img

VAL_GT = Path("/home/korvin/apps/data/val/GT")
HUB = Path(torch.hub.get_dir()) / "checkpoints"
NET_WEIGHTS = {
    "dists": [
        Path(calculate.__file__).parents[1] / "losses" / "dists_weights.pth",
        HUB / "vgg16-397923af.pth",
    ],
    "topiq": [HUB / "cfanet_fr_kadid_res50-2c4cc61d.pth"],
}

# Recorded with the implementation before the L4 changes on _synthetic_pair().
SYNTHETIC_REFERENCE = {
    "psnr": 22.927704496114075,
    "ssim": 0.412572186163428,
    "psnr_y": 27.842947491861203,
    "ssim_y": 0.5610179482616489,
}

# Recorded with the implementation before the L4 changes (GT vs bicubic x4 down/up).
REFERENCE = {
    "GT01.png": {
        "psnr": 29.38019219055825,
        "ssim": 0.7900117948192302,
        "psnr_y": 30.88443713852375,
        "ssim_y": 0.8224693809886535,
    },
    "GT02.png": {
        "psnr": 32.20234864316098,
        "ssim": 0.7462420987906807,
        "psnr_y": 33.84742978776644,
        "ssim_y": 0.7934085222881807,
    },
}
METRICS = {
    "psnr": {"type": "calculate_psnr", "crop_border": 4},
    "ssim": {"type": "calculate_ssim", "crop_border": 4},
    "psnr_y": {"type": "calculate_psnr", "crop_border": 4, "test_y_channel": True},
    "ssim_y": {"type": "calculate_ssim", "crop_border": 4, "test_y_channel": True},
}


def _pair(name: str) -> dict[str, np.ndarray]:
    gt = cv2.imread(str(VAL_GT / name))
    h, w = gt.shape[:2]
    lr = cv2.resize(gt, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    sr = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
    # same route as validation: float tensor -> tensor2img
    to_img = lambda x: tensor2img([img2tensor(x.astype(np.float32) / 255).unsqueeze(0)])
    return {"img": to_img(sr), "img2": to_img(gt)}


def _synthetic_pair() -> dict[str, np.ndarray]:
    yy, xx = np.mgrid[0:48, 0:48]
    base = np.stack([(xx * 5) % 256, (yy * 5) % 256, ((xx + yy) * 3) % 256], -1)
    noise = np.random.default_rng(0).normal(0, 20, base.shape)
    gt = np.clip(base + noise, 0, 255).astype(np.uint8)
    return {"img": cv2.GaussianBlur(gt, (5, 5), 1.2), "img2": gt}


def test_psnr_ssim_match_synthetic_reference():
    data = _synthetic_pair()
    for metric, opt in METRICS.items():
        value = calculate_metric(data, opt)
        assert value == pytest.approx(SYNTHETIC_REFERENCE[metric], rel=1e-12, abs=0)


@pytest.mark.skipif(not VAL_GT.is_dir(), reason="validation images not available")
@pytest.mark.parametrize("name", sorted(REFERENCE))
def test_psnr_ssim_match_reference(name):
    data = _pair(name)
    for metric, opt in METRICS.items():
        value = calculate_metric(data, opt)
        assert value == pytest.approx(REFERENCE[name][metric], rel=1e-12, abs=0)


def test_crop_border_validation():
    img = np.random.default_rng(0).integers(0, 256, (19, 19, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="crop_border"):
        calculate.calculate_psnr(img, img, crop_border=-1)
    with pytest.raises(ValueError, match="empty"):
        calculate.calculate_psnr(img[:8, :8], img[:8, :8], crop_border=4)
    with pytest.raises(ValueError, match="11x11"):
        calculate.calculate_ssim(img[:16, :16], img[:16, :16], crop_border=4)
    assert calculate.calculate_psnr(img[:9, :9], img[:9, :9], crop_border=4) == float("inf")
    assert calculate.calculate_ssim(img, img, crop_border=4) == pytest.approx(1.0)


class _FakeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(()))

    def forward(self, x, y):
        assert x.shape[1] == 3
        return (x - y).abs().mean() * self.w


@pytest.mark.parametrize("name", ["dists", "topiq"])
def test_metric_net_cached_per_device(monkeypatch, name):
    built = []
    monkeypatch.setitem(
        calculate._metric_builders, name, lambda: built.append(1) or _FakeNet()
    )
    release_metric_nets()
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    img2 = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    metric = getattr(calculate, f"calculate_{name}")
    try:
        first = metric(img, img2, device="cpu")
        second = metric(img, img2, device=torch.device("cpu"))
        net = calculate._metric_nets[name, torch.device("cpu")]
        assert len(built) == 1
        assert not net.training
        assert not any(p.requires_grad for p in net.parameters())
        # grayscale (what tensor2img returns for 1-channel output) is expanded to RGB
        gray = tensor2img(torch.rand(1, 1, 8, 8))
        assert type(metric(gray, gray, device="cpu")) is float
        if torch.cuda.is_available():
            metric(img, img2, device="cuda")
            metric(img, img2, device="cuda:0")
            assert len(built) == 2  # "cuda" resolves to the current device index
    finally:
        release_metric_nets()
    assert type(first) is float
    assert first == second
    assert not calculate._metric_nets


@pytest.mark.parametrize("name", ["dists", "topiq"])
def test_metric_net_real_weights(name):
    if not all(p.is_file() for p in NET_WEIGHTS[name]):
        pytest.skip(f"{name} weights not available offline")
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    img2 = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    metric = getattr(calculate, f"calculate_{name}")
    release_metric_nets()
    try:
        diff = metric(img, img2, device="cpu")
        same = metric(img, img, device="cpu")
        assert len(calculate._metric_nets) == 1
    finally:
        release_metric_nets()
    assert type(diff) is float
    if name == "dists":  # distance: 0 for identical images
        assert abs(same) < 1e-4 < diff
    else:  # similarity: identical images score higher
        assert same > diff


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_tensor2img_out_of_range(dtype):
    values = [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    tensor = torch.tensor(values, dtype=dtype).view(1, 1, 1, -1).repeat(1, 3, 1, 1)
    before = tensor.clone()
    img = tensor2img(tensor)
    assert img.dtype == np.uint8
    assert img.shape == (1, len(values), 3)
    np.testing.assert_array_equal(img[0, :, 0], [0, 0, 128, 255, 255, 255])
    assert torch.equal(tensor, before)  # caller's tensor is not clamped in place


def test_tensor2img_gray_keeps_one_channel():
    img = tensor2img(torch.full((1, 1, 4, 4), 2.0))
    assert img.shape == (4, 4)
    assert (img == 255).all()


@pytest.mark.parametrize("shape", [(1, 7), (1, 1), (5, 6)])
def test_tensor2img_2d_shapes(shape):
    assert tensor2img(torch.rand(shape)).shape == shape
