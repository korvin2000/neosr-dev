"""Wave E2 (design T2 §9): E1(c) on CUDA under AMP. One generator update is forced non-finite in both
runs (fp16: the GradScaler skips and halves its scale; bf16: the plain skip path). Run A's 6 updates
and run B's 3 + save + resume + 3 must be bit-identical, scaler and CUDA RNG included."""

import os

os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG", ":4096:8"
)  # before the first cuBLAS handle

import pytest
import torch

from neosr.runtime import Runtime
from tests.test_resume import STREAMS, assert_same, continuation, make_opt

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(autouse=True)
def deterministic():
    flags = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    yield
    torch.use_deterministic_algorithms(flags[0], warn_only=flags[1])
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = flags[2:]


def poison_next_g_update(model) -> None:
    """Make the next update's G gradient non-finite (first micro-batch of the window only)."""
    p = next(model.net_g.parameters())
    box = []

    def hook(g):
        box[0].remove()
        return torch.full_like(g, float("nan"))

    box.append(p.register_hook(hook))


@pytest.mark.parametrize("bf16", [False, True], ids=["fp16", "bf16"])
def test_cuda_amp_gan_continuation_with_a_skipped_step(tmp_path, bf16):
    def make(t):
        return make_opt(t, "c") | {"use_amp": True, "bfloat16": bf16}

    rt = Runtime(torch.device("cuda", 0), 0, 1, 0, False)
    a, r, sa, sr = continuation(tmp_path, make, acc=2, mid=poison_next_g_update, rt=rt)
    assert a.update == r.update == 6
    assert a.g.skipped == 1  # update 3 was skipped, D still stepped
    assert a.d.skipped == 0
    assert a.g.scaler.is_enabled() != bf16
    if not bf16:  # init 2**5, halved once by the skipped window
        assert a.g.scaler.get_scale() == r.g.scaler.get_scale() == 2.0**4
    assert all(sa["rng"][k] is not None for k in (*STREAMS, "cuda"))
    assert_same(
        sa, sr
    )  # each taken after its own run; the resumed build reseeded every stream
