"""Wave E3 (design T2 §9): 2-rank gloo DDP, accumulate 2 + GAN + schedule-free G, rank-specific data.
One spawn per device runs three trainers on both ranks: U (uninterrupted, rank-0 validation + save
between barriers at update 2, 4 updates), R (both ranks resume U's 2.state, 2 more updates) and S
(rank 1 alone receives SIGINT inside update 3's window). Model-side randomness is absent on purpose:
other ranks' RNG is not part of the state (T2 §11 continuation scope)."""

import os
import signal
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from neosr.models import build_model
from neosr.runtime import Runtime
from neosr.utils.stop import StopFlag
from tests.test_resume import S, ValSet, assert_same, batches, sf_x

ACC = 2
PG_TIMEOUT, DEADLINE = (
    60,
    240,
)  # s: a hung collective raises in the workers; the parent backstops


def make_opt(tmp: Path) -> dict:
    return {
        "name": "e3",
        "model_type": "image",
        "scale": S,
        "is_train": True,
        "network_g": {"type": "compact", "num_feat": 8, "num_conv": 2},
        "network_d": {"type": "unet", "num_feat": 4},  # spectral norm: rank-local buffers
        "datasets": {
            "train": {"name": "t", "type": "paired", "patch_size": 16, "batch_size": 2, "accumulate": ACC}
        },
        "val": {"pbar": False, "metrics": {"psnr": {"type": "calculate_psnr", "crop_border": 0}}},
        "logger": {},
        "path": {
            "models": tmp / "models",
            "training_states": tmp / "training_states",
            "visualization": tmp / "vis",
        },
        "train": {
            "optim_g": {"type": "adan_sf", "lr": 1e-2, "schedule_free": True},
            "optim_d": {"type": "adamw", "lr": 1e-4},
            "pixel_opt": {"type": "L1Loss", "loss_weight": 1.0},
            "gan_opt": {"type": "gan_loss", "loss_weight": 0.1},
        },
    }  # fmt: skip


def snap(m) -> dict:
    """Weights, buffers and optimizer state, on the CPU."""

    def cpu(x):
        if torch.is_tensor(x):
            return x.detach().cpu().clone()
        if isinstance(x, dict):
            return {k: cpu(v) for k, v in x.items()}
        if isinstance(x, list | tuple):
            return type(x)(cpu(v) for v in x)
        return x

    return cpu({
        "update": m.update,
        "nets": {n: net.state_dict() for n, net in m.nets.items()},
        "phases": {n: ph.state_dict() for n, ph in m.phases.items()},
    })


def step(m, data) -> None:
    for x in data:
        m.feed_data(x)
        m.optimize_parameters()


def _worker(rank: int, tmp: str, device: str, out: str) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp}/pg", rank=rank, world_size=2,
        timeout=timedelta(seconds=PG_TIMEOUT),
    )  # fmt: skip
    try:
        dev = torch.device("cuda", 0) if device == "cuda" else torch.device("cpu")
        rt = Runtime(dev, rank, 2, 0, True)
        root, res = Path(tmp), {}
        data = batches(4 * ACC, seed=rank)  # each rank sees its own micro-batches

        torch.manual_seed(rank)  # seed + rank (T3 D15): DDP must sync the start
        u = build_model(make_opt(root / "u"), rt)
        step(u, data[: 2 * ACC])
        res["u2"], res["x2"] = snap(u), sf_x(u)
        u.check_finite()  # all ranks
        u.validation(DataLoader(ValSet(), batch_size=1), 2, None, save_img=False)
        u.save()  # rank 0 writes; SF export swaps rank-0 weights and back
        dist.barrier()  # the state is on disk before any rank resumes it
        res["u2_after_val"] = snap(u)
        res["export"] = {
            n: torch.load(root / "u" / "models" / f"{n}_2.pth", weights_only=True)[
                "params"
            ]
            for n in ("net_g", "net_d")
        }
        step(u, data[2 * ACC :])
        res["u4"] = snap(u)

        opt = make_opt(root / "r")
        opt["path"]["resume_state"] = root / "u" / "training_states" / "2.state"
        torch.manual_seed(10 + rank)
        r = build_model(opt, rt)
        r.restore_rng()  # rank 0 only
        step(r, data[2 * ACC :])
        res["r4"] = snap(r)

        torch.manual_seed(rank)
        s = build_model(make_opt(root / "s"), rt)
        stop, prev = StopFlag(), signal.getsignal(signal.SIGINT)
        stop.install()
        try:
            for i, x in enumerate(data):
                if rank == 1 and i == ACC + 1:  # second micro-batch of update 2
                    os.kill(os.getpid(), signal.SIGINT)
                s.feed_data(x)
                if s.optimize_parameters() and stop.requested(dev):
                    break
        finally:
            signal.signal(signal.SIGINT, prev)
        res["stopped"] = (s.update, s.micro, stop._set)
        res["s"] = snap(s)
        torch.save(res, f"{out}.{rank}")
    finally:
        dist.destroy_process_group()


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(scope="module", params=DEVICES)
def ranks(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp(f"ddp_{request.param}")
    out = str(tmp / "res")
    ctx = torch.multiprocessing.spawn(
        _worker, args=(str(tmp), request.param, out), nprocs=2, join=False
    )
    end = time.monotonic() + DEADLINE
    while not ctx.join(timeout=1):  # raises if a worker failed
        if time.monotonic() > end:
            for p in ctx.processes:
                p.kill()
                p.join()
            pytest.fail(f"DDP workers still running after {DEADLINE} s")
    return [torch.load(f"{out}.{r}", weights_only=False) for r in (0, 1)]


def test_params_equal_across_ranks(ranks):
    r0, r1 = ranks
    for key in ("u2", "u4", "r4", "s"):
        assert_same(r0[key]["nets"], r1[key]["nets"], key)
    assert r0["u4"]["update"] == 4


def test_rank0_validation_and_export_leave_ranks_in_sync(ranks):
    r0, r1 = ranks
    for r in ranks:  # the SF swap and the eval-mode pass restore rank 0 exactly
        assert_same(r["u2"], r["u2_after_val"])
    assert_same(r0["u2_after_val"]["nets"], r1["u2_after_val"]["nets"])
    # the artifacts: SF x (an oracle from the optimizer state, not the live y) and the live D
    exp, x = r0["export"], r0["x2"]
    assert not all(torch.equal(x[k], v) for k, v in r0["u2"]["nets"]["net_g"].items())
    assert_same(exp["net_g"], x, "net_g_2")
    assert_same(exp["net_d"], r0["u2"]["nets"]["net_d"], "net_d_2")


def test_resume_on_both_ranks_equals_the_uninterrupted_run(ranks):
    r0, r1 = ranks
    assert_same(r0["r4"], r0["u4"])
    assert_same(r1["r4"]["nets"], r0["u4"]["nets"])


def test_stop_on_one_rank_stops_both_at_the_same_update(ranks):
    (u0, m0, f0), (u1, m1, f1) = (r["stopped"] for r in ranks)
    assert (f0, f1) == (False, True)  # only rank 1 got the signal
    # each boundary reads the flag reduced at the previous one (T-11): both stop one update later
    assert (u0, m0) == (u1, m1) == (3, 3 * ACC)
