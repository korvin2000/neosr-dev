"""2-rank gloo DDP on the CPU with the R3GAN recipe (r3gan D, rpgan, R1 + R2 every update) and the GAN
health log (DR-02): ranks stay bitwise in sync through the reducer, the custom-autograd conv's double
backward and the health hooks; the log on vs off leaves training bitwise the same; the health values
are measured per rank and published as the mean over ranks. Patterns: tests/test_ddp_gloo.py."""

import os
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from neosr.models import build_model
from neosr.models.image import HEALTH
from neosr.runtime import Runtime
from tests.test_ddp_gloo import snap, step
from tests.test_resume import S, assert_same, batches

ACC, UPDATES = 2, 3  # update 2 is a log update (print_freq 2)
PG_TIMEOUT, DEADLINE = (
    60,
    120,
)  # s: a hung collective raises in the workers; the parent backstops


def make_opt(tmp: Path, print_freq: int) -> dict:
    return {
        "name": "ddp_health",
        "model_type": "image",
        "scale": S,
        "is_train": True,
        "network_g": {"type": "compact", "num_feat": 8, "num_conv": 2},
        "network_d": {"type": "r3gan", "widths": [4, 8], "group_width": 2},
        "datasets": {
            "train": {"name": "t", "type": "paired", "patch_size": 16, "batch_size": 2, "accumulate": ACC}
        },
        "logger": {"print_freq": print_freq},
        "path": {
            "models": tmp / "models",
            "training_states": tmp / "training_states",
            "visualization": tmp / "vis",
        },
        "train": {
            "optim_g": {"type": "adamw", "lr": 1e-3},
            "optim_d": {"type": "adamw", "lr": 1e-4},
            "pixel_opt": {"type": "L1Loss", "loss_weight": 1.0},
            "gan_opt": {
                "type": "gan_loss", "loss_weight": 0.1, "gan_type": "rpgan",
                "r1_gamma": 1.0, "r2_gamma": 0.5, "reg_every": 1,
            },
        },
    }  # fmt: skip


def _worker(rank: int, tmp: str, out: str) -> None:
    torch.use_deterministic_algorithms(True)
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp}/pg", rank=rank, world_size=2,
        timeout=timedelta(seconds=PG_TIMEOUT),
    )  # fmt: skip
    try:
        rt = Runtime(torch.device("cpu"), rank, 2, 0, True)
        data = batches(UPDATES * ACC, seed=rank)  # each rank sees its own micro-batches
        res = {}
        for name, freq in (("on", 2), ("off", 1000)):
            torch.manual_seed(rank)  # seed + rank: DDP must sync the start
            m = build_model(make_opt(Path(tmp) / name, freq), rt)
            torch.manual_seed(
                100 + rank
            )  # the R1/R2 Rademacher draws: same in both runs
            step(m, data)
            res[name] = snap(m)
            res[name, "rng"] = torch.get_rng_state()
            # this rank's own health means (a sum over its micro-batches, before the all_reduce)
            res[name, "local"] = {k: s.item() / n for k, (s, n) in m.meter.diag.items()}
            res[name, "log"] = m.get_current_log()  # collective
        torch.save(res, f"{out}.{rank}")
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def ranks(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ddp_health")
    out = str(tmp / "res")
    env = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = "2"  # two workers share the host
    try:
        ctx = torch.multiprocessing.spawn(
            _worker, args=(str(tmp), out), nprocs=2, join=False
        )
    finally:
        if env is None:
            os.environ.pop("OMP_NUM_THREADS")
        else:
            os.environ["OMP_NUM_THREADS"] = env
    end = time.monotonic() + DEADLINE
    while not ctx.join(timeout=1):  # raises if a worker failed
        if time.monotonic() > end:
            for p in ctx.processes:
                p.kill()
                p.join()
            pytest.fail(f"DDP workers still running after {DEADLINE} s")
    return [torch.load(f"{out}.{r}", weights_only=False) for r in (0, 1)]


def test_ranks_stay_in_sync_with_r1_r2_and_the_health_log(ranks):
    r0, r1 = ranks
    assert r0["on"]["update"] == UPDATES
    assert {"r1", "r2", "l_d_reg", "out_d_margin"} <= r0[
        "on", "log"
    ].keys()  # R1/R2 ran
    for name in ("on", "off"):  # parameters and buffers
        assert_same(r0[name]["nets"], r1[name]["nets"], name)


def test_health_log_leaves_ddp_training_bitwise_unchanged(ranks):
    for r in ranks:
        assert_same(r["on"], r["off"])
        assert torch.equal(r["on", "rng"], r["off", "rng"])
        shared = r["off", "log"].keys()
        assert {k: r["on", "log"][k] for k in shared} == r["off", "log"]


def test_health_is_measured_per_rank_and_logged_as_the_rank_mean(ranks):
    r0, r1 = ranks
    assert r0["on", "local"].keys() == set(HEALTH)
    assert not HEALTH & r0["off", "log"].keys()
    for k in HEALTH:
        a, b = r0["on", "local"][k], r1["on", "local"][k]
        assert all(map(torch.isfinite, torch.tensor([a, b]))), k
        assert r0["on", "log"][k] == r1["on", "log"][k], k
        torch.testing.assert_close(r0["on", "log"][k], (a + b) / 2, msg=k)
    # different data per rank: the measurements differ before the all_reduce
    assert r0["on", "local"]["grad_sr_gan"] != r1["on", "local"]["grad_sr_gan"]
