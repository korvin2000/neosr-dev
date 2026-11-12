import logging
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Markers (registered in pyproject.toml) are derived here instead of decorating ~50 files:
# - cuda: the test needs a CUDA device. Read from what the tests already declare: a skipif whose reason
#   names CUDA (function, class, module or pytest.param level) or a "cuda" device parameter. A new CUDA
#   test is marked by the skip it needs anyway; a test that only *prefers* CUDA (DEV = "cuda" if ...) is not.
# - slow: the reviewed list SLOW below, from a measured CPU run (`--durations=0`, 2026-11-11): whole files
#   for the per-arch sweeps, node-id prefixes for single heavy tests in core files (the rest of the file stays
#   in the subset). `pytest -m "not slow and not cuda"` is the fast CPU subset (docs/refactor/STATUS.md).
# The third marker, weights(*names), is declared by the tests themselves: see _weight_files below.
SLOW = (
    # per-arch sweeps and arch lanes (CPU seconds measured on the Linux workstation)
    "tests/test_archs.py",
    "tests/test_archs_a1.py",
    "tests/test_archs_a2.py",
    "tests/test_archs_a3.py",  # 41 s
    "tests/test_s2_mixed.py",  # 90 s
    "tests/test_s2_cnn.py",  # 14 s
    "tests/test_s2_window.py",  # 11 s
    # per-arch ONNX exports
    "tests/test_s2_exportfix.py::test_export_fix_keeps_eager_outputs_bitwise",
    "tests/test_s2_exportfix.py::test_hma_dynamic_export_runs_batch_2",
    "tests/test_s2_metrics.py::test_arch_exports_",  # 24 s
    # every shipped config built with its real constructors; torch.compile; one subprocess per module
    "tests/test_losses_pipeline.py::test_shipped_configs_real_constructors",  # 71 s
    "tests/test_s2_losses.py::test_compiled_spectral_norm_d_trains",  # 19 s
    "tests/test_imports.py::test_import_has_no_side_effects",  # 19 s (~10 s per module on Windows)
)


@pytest.fixture(autouse=True)
def _caplog_sees_neosr(request):
    """caplog sees the "neosr" logger in any test order: get_root_logger sets propagate = False on
    its first call, so the logger is initialised here and propagates (to the root, where caplog's
    per-phase handlers sit) while a test that requests caplog runs."""
    if "caplog" not in request.fixturenames:
        yield
        return
    from neosr.utils.logger import get_root_logger  # noqa: PLC0415

    log = get_root_logger()
    level, prop = log.level, log.propagate
    log.propagate = True
    try:
        yield
    finally:
        log.setLevel(level)
        log.propagate = prop


def _needs_cuda(item: pytest.Item) -> bool:
    if any(
        "cuda" in str(m.kwargs.get("reason", "")).lower()
        for m in item.iter_markers("skipif")
    ):
        return True
    params = getattr(getattr(item, "callspec", None), "params", {})
    return any(str(v).startswith("cuda") for v in params.values())


def _weight_files(name: str) -> list[Path]:
    """The pretrained files behind @pytest.mark.weights(name), where the code reads them: torch hub's
    cache (torchvision weights, load_state_dict_from_url, torch.hub.load) and the DISTS file next to
    its module. Imported here, not at collection: only a test that carries the marker pays."""
    import torch  # ruff: ignore[import-outside-top-level]
    from torchvision import models  # ruff: ignore[import-outside-top-level]

    from neosr.losses import dists_loss  # ruff: ignore[import-outside-top-level]
    from neosr.metrics import lpips, topiq  # ruff: ignore[import-outside-top-level]

    hub = Path(torch.hub.get_dir())

    def cached(url: str) -> Path:
        return hub / "checkpoints" / Path(urlparse(url).path).name

    files = {
        "alexnet": [cached(models.AlexNet_Weights.IMAGENET1K_V1.url)],
        "vgg16": [cached(models.VGG16_Weights.IMAGENET1K_V1.url)],
        "vgg19": [cached(models.VGG19_Weights.DEFAULT.url)],
        "resnet18": [cached(models.ResNet18_Weights.DEFAULT.url)],
        "dists": [Path(dists_loss.__file__).with_name("dists_weights.pth")],
        "lpips_alex": [cached(lpips.URL.format(lpips.FILES["alex"]))],
        "lpips_vgg": [cached(lpips.URL.format(lpips.FILES["vgg"]))],
        # the hub copy: topiq also reads a local file first, but the tests load the hub one
        "topiq_kadid": [cached(topiq.WEIGHTS["kadid"][1])],
        # fdl_loss(model="dinov2"): torch.hub.load of the repository, which loads its checkpoint
        "dinov2": [
            hub / "facebookresearch_dinov2_main",
            hub / "checkpoints" / "dinov2_vitb14_pretrain.pth",
        ],
    }
    return files[name]


def pytest_runtest_setup(item: pytest.Item) -> None:
    """@pytest.mark.weights(*names): skip unless every file is cached; tests never download."""
    names = [n for m in item.iter_markers("weights") for n in m.args]
    missing = [f"{n} ({p})" for n in names for p in _weight_files(n) if not p.exists()]
    if missing:
        pytest.skip(f"pretrained weights not cached: {', '.join(missing)}")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if _needs_cuda(item):
            item.add_marker(pytest.mark.cuda)
        if item.nodeid.startswith(SLOW):
            item.add_marker(pytest.mark.slow)
