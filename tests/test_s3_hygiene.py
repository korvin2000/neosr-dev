"""Stage 3 A8: ONNX constant folding and the portable -optimize level, fast_matmul "high", the
detect_anomaly debug key, the dropped pytorch-optimizer dependency. CPU."""

import io
import tomllib
from pathlib import Path

import pytest
import torch

from neosr import runtime
from neosr.archs import build_network
from neosr.cli import convert_args
from neosr.config import SCHEMA
from tests.test_entry import convert_py
from tests.test_runtime import fake_gpus as fake_gpus  # noqa: F401
from tests.test_runtime import restore_torch_state as restore_torch_state  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def onnx():
    pytest.importorskip("onnxruntime")
    return pytest.importorskip("onnx")


def test_float_exports_fold_constants(tmp_path, onnx):
    """realplksr: folding precomputes its shape subgraphs (fewer nodes); verified at every shape."""
    net = build_network({"type": "realplksr", "dim": 16, "n_blocks": 2}, scale=2).eval()
    argv = ["--input", "unused.pth", "-net", "realplksr", "-s", "2"]
    convert_py.to_onnx(convert_args([*argv, "--output", str(tmp_path / "m.onnx")]), net)
    folded = len(onnx.load(str(tmp_path / "m_fp32.onnx")).graph.node)
    buf, dyn = io.BytesIO(), {"input": {0: "b", 2: "h", 3: "w"}}
    with torch.inference_mode():
        torch.onnx.export(
            net, torch.randn(1, 3, 20, 20), buf, do_constant_folding=False, dynamo=False,
            opset_version=17, input_names=["input"], dynamic_axes=dyn,
        )  # fmt: skip
    assert folded < len(onnx.load_from_string(buf.getvalue()).graph.node)


def test_optimize_writes_a_portable_graph(tmp_path, onnx):
    tiny = {"type": "compact", "num_feat": 8, "num_conv": 2}
    meta = {"network_g": tiny, "scale": 2}  # A1: the export describes itself
    torch.save(
        {"params": build_network(tiny, scale=2).state_dict(), "neosr": meta},
        tmp_path / "w.pth",
    )
    argv = ["--input", str(tmp_path / "w.pth"), "-optimize"]
    assert convert_py.main([*argv, "--output", str(tmp_path / "m.onnx")]) == 0
    graph = onnx.load(str(tmp_path / "m_fp32_optimized.onnx")).graph
    assert {n.domain for n in graph.node} <= {"", "ai.onnx"}  # no com.microsoft ops


def test_the_optimized_file_is_verified(tmp_path, onnx, monkeypatch):
    verified = []
    monkeypatch.setattr(
        convert_py, "assert_verify", lambda _a, path, *_k, **_kw: verified.append(path)
    )
    argv = ["--input", "unused.pth", "-net", "compact", "-s", "2", "-optimize"]
    net = build_network(
        {"type": "compact", "num_feat": 8, "num_conv": 2}, scale=2
    ).eval()
    convert_py.to_onnx(convert_args([*argv, "--output", str(tmp_path / "m.onnx")]), net)
    assert [Path(p).name for p in verified] == ["m_fp32.onnx", "m_fp32_optimized.onnx"]


@pytest.mark.usefixtures("restore_torch_state", "fake_gpus")
def test_fast_matmul_is_tf32_without_bf16_cpu_matmuls():
    runtime.setup_runtime({"fast_matmul": True})
    assert torch.get_float32_matmul_precision() == "high"
    assert torch.backends.cuda.matmul.allow_tf32


@pytest.fixture
def anomaly_state():
    saved = torch.is_anomaly_enabled(), torch.is_anomaly_check_nan_enabled()
    yield
    torch.autograd.set_detect_anomaly(*saved)


@pytest.mark.usefixtures("restore_torch_state", "fake_gpus", "anomaly_state")
def test_detect_anomaly_is_an_opt_in_debug_key(caplog):
    assert SCHEMA[""]["detect_anomaly"] is bool
    torch.autograd.set_detect_anomaly(False)
    runtime.setup_runtime({})
    assert not torch.is_anomaly_enabled()  # the default
    with caplog.at_level("WARNING", logger="neosr"):
        runtime.setup_runtime({"detect_anomaly": True})
    assert torch.is_anomaly_enabled()
    assert "much slower" in caplog.text
    torch.autograd.set_detect_anomaly(True, check_nan=False)  # a caller's own setting
    runtime.setup_runtime({})
    assert torch.is_anomaly_enabled()
    assert not torch.is_anomaly_check_nan_enabled()  # left alone


def test_pytorch_optimizer_is_no_dependency():
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "dependencies"
    ]
    assert not [d for d in deps if d.startswith("pytorch-optimizer")]
    assert "pytorch-optimizer" not in (ROOT / "uv.lock").read_text()
    sources = [
        *ROOT.glob("neosr/**/*.py"),
        *ROOT.glob("*.py"),
        *ROOT.glob("scripts/*.py"),
    ]
    assert not [p for p in sources if "pytorch_optimizer" in p.read_text()]
