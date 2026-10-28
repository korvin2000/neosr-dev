"""Training speed and memory of a config (stage 3 A5): steady-state ms/update, start-up and warm-up
(compile) time, peak VRAM and host RAM, eager inference ms at a given LQ size.

    python scripts/bench.py -opt options/train_realplksr.toml [--trials 3] [--updates 100] [--warmup N]
        [--compile | --eager] [--compile_mode reduce-overhead] [--precision bf16|fp16|fp32]
        [--infer 256]

Builds the trainer as train.py does (load_config, validate_train, setup_runtime, the train loader, build_model,
CUDAPrefetcher) and drives it as the loop does: feed_data + optimize_parameters, get_current_log whenever
model.update % print_freq == 0. Timing as the stage-2 harness (docs/refactor/stage2/analysis/gpu_profile.md):
an untimed warm-up (default 30 updates, 60 with compile), then `trials` x `updates` updates, each trial
bracketed by one device synchronize; ms/update is the median trial (min and max beside it). The flags
override the config's compile / compile_mode / use_amp + bfloat16. No run folder, state, export, validation
or log file is written (weights a loss needs are fetched to their caches as in training). The last line is
one JSON record. One GPU job at a time: never run it next to a training run.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neosr.config import (  # ruff: ignore[module-import-not-at-top-of-file]
    ConfigError,
    load_config,
    validate_train,
)
from neosr.data.prefetch_dataloader import CUDAPrefetcher  # ruff: ignore[module-import-not-at-top-of-file]
from neosr.models import build_model  # ruff: ignore[module-import-not-at-top-of-file]
from neosr.models.image import (  # ruff: ignore[module-import-not-at-top-of-file]
    no_cudnn_autotune,
    observers_frozen,
)
from neosr.runtime import setup_runtime  # ruff: ignore[module-import-not-at-top-of-file]
from train import build_loaders  # ruff: ignore[module-import-not-at-top-of-file]

PRECISION = {  # flag -> (use_amp, bfloat16)
    "bf16": (True, True),
    "fp16": (True, False),
    "fp32": (False, False),
}


def parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="neosr training benchmark")
    p.add_argument("-opt", required=True, help="training TOML")
    p.add_argument("--trials", type=int, default=3, help="timed trials (default 3)")
    p.add_argument(
        "--updates", type=int, default=100, help="updates per trial (default 100)"
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="untimed updates first, >= 1 (default 30, 60 with compile)",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--compile", dest="compile", action="store_true", default=None)
    mode.add_argument("--eager", dest="compile", action="store_false")
    p.add_argument(
        "--compile_mode",
        default=None,
        help="torch.compile mode, e.g. reduce-overhead, max-autotune",
    )
    p.add_argument("--precision", choices=sorted(PRECISION), default=None)
    p.add_argument(
        "--infer",
        type=int,
        default=256,
        help="LQ size of the inference timing; 0 skips it",
    )
    args = p.parse_args(argv)
    if (
        min(args.trials, args.updates, 1 if args.warmup is None else args.warmup) < 1
        or args.infer < 0
    ):
        p.error("--trials, --updates and --warmup must be >= 1, --infer >= 0")
    return args


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _host_peak_mib() -> float | None:
    """Peak RSS of this process (loader workers apart); None where `resource` is missing (Windows)."""
    try:
        import resource  # noqa: PLC0415
    except ImportError:
        return None
    return round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
    )  # KiB on Linux


def _vram_mib(dev: torch.device) -> float | None:
    if dev.type != "cuda":
        return None
    peak = torch.cuda.max_memory_allocated(dev) / 2**20
    torch.cuda.reset_peak_memory_stats(dev)
    return round(peak, 1)


def _precision(cfg: dict) -> str:
    amp = (bool(cfg.get("use_amp")), bool(cfg.get("bfloat16")))
    return next(k for k, v in PRECISION.items() if v == amp)


def _updates(model, sampler, prefetcher, print_freq: int):
    """Endless completed updates, epoch after epoch, with the loop's log reads."""
    epoch = 0
    while True:
        model.epoch = epoch
        sampler.set_epoch(epoch)
        prefetcher.reset()
        while (batch := prefetcher.next()) is not None:
            model.feed_data(batch)
            if model.optimize_parameters():
                if model.update % print_freq == 0:
                    model.get_current_log()  # one collective, one host sync, as train.py
                yield
        epoch += 1


def bench(cfg: dict, args: argparse.Namespace) -> dict:
    t0 = time.perf_counter()
    rt = setup_runtime(cfg)
    dev = rt.device
    loader, sampler, _ = build_loaders(cfg, rt)
    model = build_model(cfg, rt)
    model.data_generator = loader.generator
    steps = _updates(
        model,
        sampler,
        CUDAPrefetcher(loader, dev),
        cfg["logger"].get("print_freq", 100),
    )
    _sync(dev)
    out = {
        "config": cfg["opt_path"],
        "device": str(dev),
        "compile": bool(cfg.get("compile")),
        "compile_mode": cfg.get("compile_mode", "default"),
        "precision": _precision(cfg),
        "build_s": round(time.perf_counter() - t0, 3),
    }
    warmup = args.warmup if args.warmup is not None else 60 if out["compile"] else 30
    t = time.perf_counter()
    next(
        steps
    )  # the first update: compilation of the step's graphs, cuDNN autotuning, allocations
    _sync(dev)
    out["first_update_s"] = round(time.perf_counter() - t, 3)
    t = time.perf_counter()
    for _ in range(warmup - 1):
        next(steps)
    _sync(dev)
    out |= {"warmup_updates": warmup, "warmup_s": round(time.perf_counter() - t, 3)}
    out["peak_vram_warmup_mib"] = _vram_mib(
        dev
    )  # start-up + warm-up (compile, autotuning)
    trials = []
    for _ in range(args.trials):
        t = time.perf_counter()
        for _ in range(args.updates):
            next(steps)
        _sync(dev)
        trials.append((time.perf_counter() - t) * 1e3 / args.updates)
    out |= {
        "trials": args.trials,
        "updates_per_trial": args.updates,
        "ms_per_update": round(statistics.median(trials), 3),
        "ms_per_update_min_max": [round(min(trials), 3), round(max(trials), 3)],
        "peak_vram_train_mib": _vram_mib(dev),  # the timed trials
    }
    if args.infer > 0:
        out |= _infer(model, cfg, dev, args.infer)
    out["peak_host_rss_mib"] = _host_peak_mib()
    return out


def _infer(model, cfg: dict, dev: torch.device, size: int, runs: int = 10) -> dict:
    """The evaluated generator (EMA / SF x), eager, under the contexts of validate() (eval mode, QAT
    ranges frozen, cuDNN autotuning off) and the run's precision as test.py applies it."""
    amp = bool(cfg.get("use_amp")) and dev.type == "cuda"
    dtype = torch.bfloat16 if cfg.get("bfloat16") else torch.float16
    x = torch.rand(1, 3, size, size, device=dev)
    times = []
    with (
        model.eval_weights() as net,
        observers_frozen(net),
        no_cudnn_autotune(),
        torch.inference_mode(),
    ):
        for i in range(runs + 3):  # 3 untimed: allocations and kernel selection
            _sync(dev)
            t = time.perf_counter()
            with torch.autocast(dev.type, dtype, enabled=amp):
                net(x)
            _sync(dev)
            if i >= 3:
                times.append((time.perf_counter() - t) * 1e3)
    return {
        "infer_size": size,
        "infer_ms": round(statistics.median(times), 3),
        "peak_vram_infer_mib": _vram_mib(dev),
    }


def main(argv=None) -> int:
    args = parse(argv)
    try:
        cfg = load_config(args.opt, is_train=True, root=ROOT)
        # train datasets only; nothing resumes, validates or writes
        cfg["datasets"] = {k: v for k, v in cfg["datasets"].items() if k == "train"}
        cfg.pop("val", None)
        cfg["path"]["resume_state"] = None
        if args.compile is not None:
            cfg["compile"] = args.compile
        if args.compile_mode is not None:
            cfg["compile_mode"] = args.compile_mode
        if args.precision is not None:
            cfg["use_amp"], cfg["bfloat16"] = PRECISION[args.precision]
        validate_train(cfg)
        out = bench(cfg, args)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    for k, v in out.items():
        print(f"{k:>24}: {v}")
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    sys.exit(main())
