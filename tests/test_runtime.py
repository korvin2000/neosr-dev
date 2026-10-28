import dataclasses
import json
import os
import random
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from neosr import runtime
from neosr.config import ConfigError
from neosr.utils import dist_util

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def restore_torch_state(monkeypatch: pytest.MonkeyPatch):
    """setup_runtime sets process-wide flags, seeds and environment variables: put them back."""
    for var in (
        "CUBLAS_WORKSPACE_CONFIG",
        "PYTORCH_ALLOC_CONF",
        "PYTORCH_CUDA_ALLOC_CONF",
    ):
        monkeypatch.setenv(var, os.environ.get(var, ""))  # registers the restore
        monkeypatch.delenv(var)
    cudnn, matmul = torch.backends.cudnn, torch.backends.cuda.matmul
    flags = (
        cudnn.deterministic,
        cudnn.benchmark,
        cudnn.benchmark_limit,
        cudnn.allow_tf32,
        matmul.allow_tf32,
        matmul.allow_fp16_accumulation,
        torch.get_float32_matmul_precision(),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.utils.deterministic.fill_uninitialized_memory,
    )
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())  # noqa: NPY002
    yield
    (
        cudnn.deterministic,
        cudnn.benchmark,
        cudnn.benchmark_limit,
        cudnn.allow_tf32,
        matmul.allow_tf32,
        matmul.allow_fp16_accumulation,
        precision,
        det,
        warn_only,
        torch.utils.deterministic.fill_uninitialized_memory,
    ) = flags
    torch.set_float32_matmul_precision(precision)
    torch.use_deterministic_algorithms(det, warn_only=warn_only)
    random.setstate(rng[0])
    np.random.set_state(rng[1])  # noqa: NPY002
    torch.set_rng_state(rng[2])


@pytest.fixture
def fake_gpus(monkeypatch: pytest.MonkeyPatch):
    """Two fake CUDA devices of compute capability 8.9; no real CUDA call is made."""
    calls: list = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (8, 9))
    monkeypatch.setattr(torch.cuda, "set_device", calls.append)
    return calls


@pytest.mark.usefixtures("restore_torch_state", "fake_gpus")
def test_single_process() -> None:
    cfg = {
        "manual_seed": 7,
        "deterministic": True,
        "fast_matmul": True,
        "num_gpu": "auto",
    }
    rt = runtime.setup_runtime(cfg)
    assert rt == runtime.Runtime(torch.device("cuda", 0), 0, 1, 7, distributed=False)
    recorded = {"dist": False, "rank": 0, "world_size": 1, "manual_seed": 7}
    assert cfg | recorded | {"deterministic": True, "num_gpu": 2} == cfg
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark
    assert torch.are_deterministic_algorithms_enabled()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.backends.cuda.matmul.allow_tf32
    assert torch.get_float32_matmul_precision() == "medium"  # no mixed-API RuntimeError
    assert random.random() == random.Random(7).random()
    with pytest.raises(dataclasses.FrozenInstanceError):
        rt.seed = 1  # type: ignore[misc]


@pytest.mark.usefixtures("restore_torch_state", "fake_gpus")
def test_random_seed_without_manual_seed() -> None:
    cfg: dict = {}
    rt = runtime.setup_runtime(cfg)
    assert 1024 <= rt.seed <= 10000
    assert (cfg["manual_seed"], cfg["deterministic"]) == (rt.seed, False)
    assert torch.backends.cudnn.benchmark
    assert not torch.are_deterministic_algorithms_enabled()
    assert os.environ["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"


@pytest.mark.usefixtures("restore_torch_state", "fake_gpus")
def test_launcher_uses_local_rank_and_dist_params(monkeypatch) -> None:
    init: list = []
    monkeypatch.setattr(
        runtime, "init_dist", lambda launcher, **kw: init.append((launcher, kw))
    )
    # global rank 2 and local rank 1 pick different GPUs out of two
    monkeypatch.setattr(runtime, "get_dist_info", lambda: (2, 4))
    monkeypatch.setenv("LOCAL_RANK", "1")
    params = {"backend": "gloo", "port": 29501, "backed": "nccl"}
    cfg = {"manual_seed": 5, "dist_params": params}
    rt = runtime.setup_runtime(cfg, launcher="slurm")
    assert init == [
        ("slurm", {"backend": "gloo", "port": 29501})
    ]  # unknown keys dropped
    assert rt == runtime.Runtime(torch.device("cuda", 1), 2, 4, 5, distributed=True)
    assert cfg["dist"] is rt.distributed
    assert random.random() == random.Random(5 + 2).random()  # seeded with seed + rank


@pytest.mark.usefixtures("restore_torch_state", "fake_gpus")
def test_explicit_multi_gpu_needs_a_launcher() -> None:
    with pytest.raises(ConfigError, match="num_gpu=2 needs a launcher"):
        runtime.setup_runtime({"manual_seed": 1, "num_gpu": 2})
    cfg = {"manual_seed": 1, "num_gpu": "auto"}
    assert runtime.setup_runtime(cfg).device == torch.device("cuda", 0)
    assert cfg["num_gpu"] == 2


@pytest.mark.usefixtures("restore_torch_state", "fake_gpus")
def test_bfloat16_on_pascal_is_a_config_error(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (6, 1))
    with pytest.raises(ConfigError, match="bfloat16"):
        runtime.setup_runtime({"manual_seed": 1, "use_amp": True, "bfloat16": True})


def test_pytorch_launcher_reads_local_rank(fake_gpus, monkeypatch) -> None:
    monkeypatch.setattr(dist_util.dist, "is_initialized", lambda: True)
    monkeypatch.setenv("RANK", "2")  # the global rank would pick GPU 0 of two
    monkeypatch.setenv("LOCAL_RANK", "1")
    dist_util._init_dist_pytorch("nccl")
    assert fake_gpus == [1]


def test_slurm_receives_port(monkeypatch) -> None:
    calls: list = []
    monkeypatch.setattr(dist_util, "_init_dist_slurm", lambda *a: calls.append(a))
    dist_util.init_dist("slurm", backend="gloo", port=1234)
    assert calls == [("gloo", 1234)]
    with pytest.raises(ValueError, match="Invalid launcher"):
        dist_util.init_dist("mpi")


# Two CPU ranks over gloo. The real `neosr/__init__` still parses argv at import, so it is stubbed (see
# test_imports.py); CUDA is hidden so no GPU is touched.
RANK_SCRIPT = """
import json, os, sys, types
stub = types.ModuleType("neosr"); stub.__path__ = [os.path.join(os.environ["REPO"], "neosr")]
sys.modules["neosr"] = stub
import torch.distributed as dist
from neosr.runtime import setup_runtime
cfg = {"dist_params": {"backend": "gloo"}}
rt = setup_runtime(cfg, launcher="pytorch")
print(json.dumps({"seed": rt.seed, "rank": rt.rank, "world": rt.world_size, "dist": cfg["dist"],
                  "distributed": rt.distributed, "device": str(rt.device), "cfg_seed": cfg["manual_seed"]}))
dist.destroy_process_group()
"""


def test_two_rank_gloo_shares_rank0_seed(tmp_path: Path) -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    base = {
        **os.environ,
        "REPO": str(ROOT),
        "PYTHONPATH": str(ROOT),
        "CUDA_VISIBLE_DEVICES": "",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
        "WORLD_SIZE": "2",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", RANK_SCRIPT],
            cwd=tmp_path,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**base, "RANK": str(r), "LOCAL_RANK": str(r)},
        )
        for r in range(2)
    ]
    outs = [p.communicate(timeout=120) for p in procs]
    assert all(p.returncode == 0 for p in procs), [err[-2000:] for _, err in outs]
    infos = [json.loads(out.strip().splitlines()[-1]) for out, _ in outs]
    assert [i["rank"] for i in infos] == [0, 1]
    assert (
        infos[0]["seed"]
        == infos[1]["seed"]
        == infos[0]["cfg_seed"]
        == infos[1]["cfg_seed"]
    )
    for info in infos:
        assert info["world"] == 2
        assert info["dist"] is info["distributed"] is True
        assert info["device"] == "cpu"
