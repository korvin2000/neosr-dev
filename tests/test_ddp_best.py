"""val.save_best under 2-rank gloo DDP on the CPU (STATUS "Needs the Linux machine" item 9): rank 0
validates and alone writes the best export, once per improvement; no rank waits at the validation
barriers; a failing best export raises on every rank and records nothing; a resume on both ranks
restores rank 0's best record. Patterns: tests/test_ddp_gloo.py, test_s2_trainer.py (T-13)."""

import copy
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from neosr.models import base as base_mod
from neosr.runtime import Runtime
from tests.test_ddp_gloo import snap, step
from tests.test_model import batches, build, loader, make_opt
from tests.test_resume import assert_same

# scripted PSNR per validation update: improvements (>=) at 1, 2, 4; 3 is worse
SCRIPT = {1: 10.0, 2: 12.0, 3: 11.0, 4: 13.0, 5: 20.0}
BEST = "net_g_best_valset_psnr.pth"
DEADLINE = 120  # s: the parent backstops a hung worker


def _opt(tmp: Path) -> dict:
    opt = make_opt(tmp)
    opt["val"]["save_best"] = "psnr"
    return opt


def _worker(rank: int, tmp: str, out: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp}/pg", rank=rank, world_size=2,
        timeout=timedelta(seconds=60),
    )  # fmt: skip
    try:
        rt, root, res = Runtime(torch.device("cpu"), rank, 2, 0, True), Path(tmp), {}
        writes: list[tuple[str, float | None]] = []
        real_save = base_mod.atomic_save

        def spy(obj, path):  # every file the model writes, with the best export's value
            writes.append((
                Path(path).name,
                obj.get("neosr", {}).get("metric", {}).get("value"),
            ))
            real_save(obj, path)

        base_mod.atomic_save = spy
        m = build(_opt(root / "run"), rt=rt, seed=rank)  # DDP syncs the start
        orig = m.update_best
        m.update_best = lambda ds, k, _v, it: orig(
            ds, k, SCRIPT[it], it
        )  # real PSNR, scripted
        data = batches(4, seed=rank)
        t = time.monotonic()
        for u in range(1, 5):
            step(m, data[u - 1 : u])
            m.validation(loader(), u, None, save_img=False)
            res[u] = m.update
        res["seconds"] = time.monotonic() - t
        m.save()  # rank 0 writes 4.state with the best record
        dist.barrier()
        res["writes"], res["best"] = list(writes), copy.deepcopy(m.best_metric_results)

        # a failing best export (rank 0): every rank raises, the record stays as it was
        def full(obj, path):
            if Path(path).name == BEST:
                raise OSError("disk full")
            real_save(obj, path)

        base_mod.atomic_save = full
        t = time.monotonic()
        try:
            m.validation(loader(), 5, None, save_img=False)
        except Exception as e:  # noqa: BLE001
            res["raised"] = f"{type(e).__name__}: {e}"
        res["fail_seconds"] = time.monotonic() - t
        res["best_after_fail"] = m.best_metric_results
        base_mod.atomic_save = real_save

        opt = _opt(root / "resumed")
        opt["path"]["resume_state"] = root / "run" / "training_states" / "4.state"
        r = build(opt, rt=rt, seed=10 + rank)
        res["resumed_best"], res["resumed"], res["run"] = (
            r.best_metric_results,
            snap(r),
            snap(m),
        )
        torch.save(res, f"{out}.{rank}")
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def ranks(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ddp_best")
    out = str(tmp / "res")
    ctx = torch.multiprocessing.spawn(
        _worker, args=(str(tmp), out), nprocs=2, join=False
    )
    end = time.monotonic() + DEADLINE
    while not ctx.join(timeout=1):  # raises if a worker failed
        if time.monotonic() > end:
            for p in ctx.processes:
                p.kill()
                p.join()
            pytest.fail(f"DDP workers still running after {DEADLINE} s")
    return [torch.load(f"{out}.{r}", weights_only=False) for r in (0, 1)]


def test_rank0_alone_writes_the_best_export_once_per_improvement(ranks):
    r0, r1 = ranks
    best = [v for name, v in r0["writes"] if name == BEST]
    assert best == [10.0, 12.0, 13.0]  # updates 1, 2, 4; not 3
    assert r1["writes"] == []
    assert r0["best"]["valset"]["psnr"] == {"better": "higher", "val": 13.0, "iter": 4}
    for u in range(1, 5):  # both ranks reach every validation at the same update
        assert r0[u] == r1[u] == u
    # the record lives on rank 0 (rank 0 validates and saves the state); rank 1 holds none
    assert r1["best"] == {}
    # nobody waited out a barrier timeout
    assert max(r0["seconds"], r1["seconds"]) < 60


def test_failed_best_export_raises_everywhere_and_records_nothing(ranks):
    r0, r1 = ranks
    assert r0["raised"] == "OSError: disk full"
    assert r1["raised"] == "RuntimeError: validation failed on rank 0"
    assert r1["fail_seconds"] < 30  # not stranded until the process-group timeout
    assert r0["best_after_fail"] == r0["best"]  # 20.0 @ 5 was rolled back


def test_resume_restores_rank0s_best_record_on_both_ranks(ranks):
    r0, r1 = ranks
    for r in ranks:
        assert r["resumed_best"] == r0["best"]
    assert_same(r0["resumed"]["nets"], r1["resumed"]["nets"])
    assert_same(r0["resumed"]["nets"], r0["run"]["nets"])
