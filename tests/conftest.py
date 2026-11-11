import logging
import sys
from pathlib import Path

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


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if _needs_cuda(item):
            item.add_marker(pytest.mark.cuda)
        if item.nodeid.startswith(SLOW):
            item.add_marker(pytest.mark.slow)
