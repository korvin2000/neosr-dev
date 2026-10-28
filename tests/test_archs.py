"""L-A acceptance (T3 §5), one case per ARCHS row; select a batch with `-k <stem>` (e.g. `-k dat_arch`).

Equivalence compares against the pre-migration fixture written by the old code (T3 §5 phase 0,
`arch_equivalence.py`); point NEOSR_ARCH_FIXTURE at its directory, else those checks skip.
Once it is set, a missing fixture fails (rows that cannot run are xfailed in `cases`).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import Tensor

from neosr.archs import ARCHS, build_network

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(d) if (d := os.environ.get("NEOSR_ARCH_FIXTURE")) else None
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

EXTRA = {"VGGFeatureExtractor": {"layer_name_list": ["relu1_1", "relu2_1", "relu3_1"]}}
SIZE = {
    "lmlt": 64,
    "lmlt_tiny": 64,
    "lmlt_large": 64,
    "r3gan_scalar256": 256,
}  # lmlt pads to its window only above 32 px; the r3gan scalar head needs an exact 4x4 final map
UNSUPPORTED = {("swinir_large", 2): "swinir_large asserts x4 in its ctor"}
ONLY_1X = {"nafnet", "nafnet_l"}  # 1x restoration networks
# added after the pre-migration fixture archive: smoke-tested only in NEOSR_ARCH_FIXTURE mode
NO_FIXTURE = {"nafnet", "nafnet_l", "r3gan", "r3gan_scalar128", "r3gan_scalar256"}
TOO_BIG = {
    "hit_srf_large": "allocates >9.6 GiB per forward at any input size (shared 24 GiB GPU)"
}
TRAIN_TOO_BIG = {
    "hit_srf_medium": "train forward needs >20 GiB at any input size (position-bias tensor)"
}


def cases() -> list:
    out = []
    for name, (stem, kw) in ARCHS.items():
        for scale in (1,) if name in ONLY_1X else (2, 4) if kw else (None,):
            marks = []
            if (name, scale) in UNSUPPORTED:
                marks.append(
                    pytest.mark.xfail(
                        raises=AssertionError,
                        strict=True,
                        reason=UNSUPPORTED[name, scale],
                    )
                )
            if name in TOO_BIG:
                marks.append(pytest.mark.xfail(run=False, reason=TOO_BIG[name]))
            out.append(
                pytest.param(
                    name, scale, marks=marks, id=f"{stem}-{name}-x{scale or 0}"
                )
            )
    return out


def tensors(out: Tensor | dict | list | tuple) -> list[Tensor]:
    if isinstance(out, Tensor):
        return [out]
    return [
        t for v in (out.values() if isinstance(out, dict) else out) for t in tensors(v)
    ]


@pytest.fixture
def _exact():
    """The fixture's flags: no TF32, deterministic kernels. Archs must not need a default CUDA device."""
    assert torch.get_default_device().type == "cpu"
    b = torch.backends
    saved = (
        b.cuda.matmul.allow_tf32,
        b.cudnn.allow_tf32,
        b.cudnn.benchmark,
        b.cudnn.deterministic,
    )
    det = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    b.cuda.matmul.allow_tf32 = b.cudnn.allow_tf32 = b.cudnn.benchmark = False
    b.cudnn.deterministic = True
    torch.use_deterministic_algorithms(mode=True, warn_only=True)
    yield
    (
        b.cuda.matmul.allow_tf32,
        b.cudnn.allow_tf32,
        b.cudnn.benchmark,
        b.cudnn.deterministic,
    ) = saved
    torch.use_deterministic_algorithms(det[0], warn_only=det[1])
    torch.cuda.empty_cache()


IMPORT_PROBE = """
import importlib, json, os, sys, types
sys.argv = ["x"]
for name in ("neosr", "neosr.archs"):  # measure the arch module alone, not the package star-imports
    stub = types.ModuleType(name)
    stub.__path__ = [os.path.join(os.environ["PROBE_REPO"], *name.split("."))]
    sys.modules[name] = stub
res = {}
for stem in os.environ["PROBE_STEMS"].split(","):
    try:
        importlib.import_module("neosr.archs." + stem)
        res[stem] = None
    except BaseException as e:
        res[stem] = f"{type(e).__name__}: {e}"
import torch
res["_cuda"] = torch.cuda.is_initialized()
print(json.dumps(res))
"""


@pytest.fixture(scope="module")
def imports(tmp_path_factory) -> dict:
    stems = sorted({stem for stem, _ in ARCHS.values()})
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PROBE_REPO": str(ROOT),
        "PROBE_STEMS": ",".join(stems),
    }
    r = subprocess.run(
        [sys.executable, "-c", IMPORT_PROBE],
        cwd=tmp_path_factory.mktemp("probe"),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("stem", sorted({stem for stem, _ in ARCHS.values()}))
def test_import_without_opt(stem: str, imports: dict) -> None:
    assert imports[stem] is None, imports[stem]
    assert imports["_cuda"] is False


@cuda
@pytest.mark.usefixtures("_exact")
@pytest.mark.parametrize(("name", "scale"), cases())
def test_arch(name: str, scale: int | None) -> None:
    kw = ARCHS[name][1]
    torch.manual_seed(0)
    cfg = {"type": name, **EXTRA.get(name, {})}
    net = (
        build_network(cfg, scale=scale)
        if kw
        else build_network(cfg, section="network_d")
    )
    assert torch.get_default_device().type == "cpu"
    net = net.cuda()
    size = SIZE.get(name, 32 if kw else 128)

    fx = None
    if FIXTURE is not None and name not in NO_FIXTURE:
        fx = torch.load(FIXTURE / f"{name}_s{scale or 4}.pt", weights_only=True)
        net.load_state_dict(fx["state_dict"], strict=True)
    x = fx["input"] if fx else torch.rand(1, 3, size, size)
    assert x.shape[-1] == size
    with torch.no_grad():
        out = tensors(net.eval()(x.cuda()))
    if kw:
        assert out[0].shape[-2:] == (size * scale, size * scale)
    if fx:
        for o, ref in zip(out, tensors(fx["output"]), strict=True):
            torch.testing.assert_close(o.cpu(), ref, atol=1e-6, rtol=0)

    if name in TRAIN_TOO_BIG:
        pytest.xfail(TRAIN_TOO_BIG[name])
    net.train()
    x = torch.rand(2, 3, size, size, device="cuda", requires_grad=True)
    out = tensors(net(x))
    if kw:
        assert out[0].shape[-2:] == (size * scale, size * scale)
    sum(o.float().mean() for o in out).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    grads = [p.grad for p in net.parameters() if p.requires_grad and p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)
    assert grads or not any(p.requires_grad for p in net.parameters())
    if fx is None:
        pytest.skip("equivalence needs NEOSR_ARCH_FIXTURE")


def test_metagan_blocks_get_distinct_weights() -> None:
    torch.manual_seed(0)
    net = build_network({"type": "metagan"}, section="network_d")
    for stage in net.stages[:4]:
        w = [blk.fc1.weight for blk in stage.blocks]
        assert all(not torch.equal(a, b) for i, a in enumerate(w) for b in w[i + 1 :])
    torch.manual_seed(0)
    again = build_network({"type": "metagan"}, section="network_d")
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            net.state_dict().values(), again.state_dict().values(), strict=True
        )
    )
