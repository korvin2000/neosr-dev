"""Training entry point (design T3 §3.7, T2 §5.1). Library code raises; only main() maps exceptions
to exit codes: 0 done or stopped by Ctrl-C, 1 unexpected (traceback logged), 2 ConfigError, 130 a
second Ctrl-C (the last committed checkpoint stands)."""

import datetime
import logging
import signal
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from neosr.cli import train_args
from neosr.config import ConfigError, load_config, validate_train
from neosr.data import build_dataloader, build_dataset
from neosr.data.data_sampler import EnlargedSampler
from neosr.data.prefetch_dataloader import CUDAPrefetcher
from neosr.models import build_model
from neosr.models.checkpoint import latest_state
from neosr.models.image import observers_frozen
from neosr.runtime import Runtime, setup_runtime
from neosr.utils import (
    AvgTimer,
    MessageLogger,
    get_root_logger,
    get_time_str,
    init_tb_logger,
    init_wandb_logger,
    make_exp_dirs,
    mkdir_and_rename,
)
from neosr.utils.dist_util import master_only
from neosr.utils.stop import StopFlag

ROOT = Path(__file__).resolve().parent
log = logging.getLogger("neosr")


@master_only
def save_config_copy(src: str | Path, dst_dir: str | Path, argv: Sequence[str]) -> None:
    """Copy the TOML into the run folder with a provenance header (T3 D27)."""
    src = Path(src)
    dst = Path(dst_dir) / f"{src.stem}_{get_time_str()}{src.suffix}"
    header = f"# GENERATE TIME: {time.asctime()}\n# CMD:\n# {' '.join(argv)}\n\n"
    dst.write_text(header + src.read_text(encoding="utf-8"), encoding="utf-8")


def select_resume(cfg: dict[str, Any]) -> Path | None:
    """T2 D17: an explicit path.resume_state wins and must be a file; else auto_resume takes the
    newest <update>.state. The choice is stored in path.resume_state for the model."""
    p = cfg["path"]
    if explicit := p.get("resume_state"):
        resume = Path(explicit)
        if not resume.is_absolute():
            resume = Path(cfg["root_path"]) / resume
        if not resume.is_file():
            msg = f"resume_state {resume} does not exist"
            raise FileNotFoundError(msg)
    elif cfg.get("auto_resume"):
        resume = latest_state(Path(p["training_states"]))
    else:
        resume = None
    p["resume_state"] = resume
    return resume


def prepare_run(cfg: dict[str, Any], argv: Sequence[str]) -> None:
    """Resume selection, run folders (a fresh run archives an existing one; a resume never does),
    config copy and the log file."""
    p = cfg["path"]
    # a resumed state may come from elsewhere: its run folders need not exist yet
    make_exp_dirs(cfg, archive=select_resume(cfg) is None)
    save_config_copy(cfg["opt_path"], p["experiments_root"], argv)
    get_root_logger(
        log_file=str(Path(p["log"]) / f"train_{cfg['name']}_{get_time_str()}.log")
    )
    if p["resume_state"] is not None:
        log.info(f"resuming from {p['resume_state']}")


def init_tb_loggers(cfg: dict[str, Any], rt: Runtime, fresh: bool) -> Any:
    lg, name = cfg["logger"], cfg["name"]
    if rt.rank != 0 or "debug" in name:
        return None
    if (lg.get("wandb") or {}).get("project") is not None:
        if not lg.get("use_tb_logger"):
            msg = "logger.wandb syncs the tensorboard log: enable logger.use_tb_logger"
            raise ConfigError(msg)
        init_wandb_logger(cfg)
    if not lg.get("use_tb_logger"):
        return None
    tb_dir = Path(cfg["path"]["experiments_root"]).parent / "tb_logger" / name
    if fresh:
        mkdir_and_rename(str(tb_dir))
    return init_tb_logger(log_dir=str(tb_dir))


def build_loaders(
    cfg: dict[str, Any], rt: Runtime
) -> tuple[DataLoader, EnlargedSampler, list[DataLoader]]:
    """Train loader + sampler and the val loaders; empty loaders raise ConfigError (T2 D27)."""
    loader = sampler = None
    val_loaders = []
    for phase, ds_cfg in cfg["datasets"].items():
        kind = phase.split("_")[0]
        if kind not in {"train", "val"}:
            msg = f"[datasets.{phase}]: training configs take train and val* datasets"
            raise ConfigError(msg)
        ds = build_dataset(ds_cfg)
        if kind == "val":
            val_loaders.append(vl := build_dataloader(ds, ds_cfg, runtime=rt))
            if len(vl) == 0:
                msg = f"[datasets.{phase}] has no images"
                raise ConfigError(msg)
            log.info(f"validation set {ds_cfg.get('name', phase)}: {len(vl)} images")
            continue
        ratio = ds_cfg.get("dataset_enlarge_ratio", 1)
        sampler = EnlargedSampler(ds, rt.world_size, rt.rank, ratio, seed=rt.seed)
        loader = build_dataloader(ds, ds_cfg, runtime=rt, sampler=sampler)
        if len(loader) == 0:
            msg = (
                f"[datasets.train] yields no batch: {len(ds)} images x enlarge ratio {ratio} "  # type: ignore[arg-type]
                f"over {rt.world_size} process(es) is less than batch_size {ds_cfg['batch_size']}"
            )
            raise ConfigError(msg)
        acc = ds_cfg.get("accumulate", 1)
        log.info(
            "Training information:"
            f"\n-------- Model: {cfg['name']}"
            f"\n-------- Processes: {rt.world_size}, device: {rt.device}"
            f"\n-------- Patch size: {ds_cfg['patch_size']}"
            f"\n-------- Dataset size: {len(ds)}"  # type: ignore[arg-type]
            f"\n-------- Batch size per process: {ds_cfg['batch_size']}, accumulate: {acc}"
            f"\n-------- Updates per epoch: {len(loader) / acc:.1f}"
            f"\n-------- Total updates: {cfg['logger'].get('total_iter', 1000000)}"
        )
    if loader is None or sampler is None:
        msg = "training needs a [datasets.train] section"
        raise ConfigError(msg)
    return loader, sampler, val_loaders


def log_macs(model: Any, cfg: dict[str, Any]) -> None:
    """T2 D28: bare nets, eval mode, no_grad, before restore_rng(); per-module train flags restored.
    Only when `input_size` is set; torchprofile is optional."""
    size = cfg.get("input_size")
    if size is None:
        return
    try:
        from torchprofile import profile_macs  # noqa: PLC0415
    except ImportError:
        log.warning(
            "input_size is set but torchprofile is not installed: MACs not reported"
        )
        return
    g = torch.Generator().manual_seed(0)  # leaves the run's RNG streams alone
    x = torch.rand(1, cfg.get("in_chans", 3), size, size, generator=g).to(model.device)
    with torch.no_grad():
        for label, net in (("Generator", model.net_g), ("Discriminator", model.net_d)):
            if net is None:
                continue
            modes = [(m, m.training) for m in net.modules()]
            net.eval()
            try:
                with observers_frozen(net):  # eval() leaves QAT observers running
                    macs = profile_macs(net, x)
                    if net is model.net_g:
                        x = net(x)  # D sees what G produces
            finally:
                for m, t in modes:
                    m.training = t
            log.info(f"{label} MACs: {macs:,d}")


def _freq(lg: dict[str, Any], key: str, default: int | None) -> int:
    val = lg.get(key, default)
    if val is None or val < 1:
        msg = f"logger.{key} must be set to a positive number of updates"
        raise ConfigError(msg)
    return val


def train(cfg: dict[str, Any], rt: Runtime) -> None:
    """T2 §5.1. Call prepare_run() first. Every frequency counts completed updates (model.update)."""
    lg, val = cfg.setdefault("logger", {}), cfg.get("val") or {}
    total = _freq(lg, "total_iter", 1000000)
    print_freq = _freq(lg, "print_freq", 100)
    save_freq = _freq(lg, "save_checkpoint_freq", None)
    val_freq = val.get("val_freq")
    save_img = val.get("save_img", True)

    gpu = (
        f" ({torch.cuda.get_device_name(rt.device)})"
        if rt.device.type == "cuda"
        else ""
    )
    log.info(f"PyTorch {torch.__version__}, device {rt.device}{gpu}")
    model = build_model(cfg, rt)  # resumes inside (T2 §3 step 7)
    loader, sampler, val_loaders = build_loaders(cfg, rt)
    model.data_generator = loader.generator  # part of the RNG state (T3 D18)
    log_macs(model, cfg)
    if model.update >= total:
        log.info(f"nothing to do: update {model.update} >= total_iter {total}")
        return
    tb_logger = init_tb_loggers(cfg, rt, fresh=cfg["path"]["resume_state"] is None)
    msg_logger = MessageLogger(cfg, tb_logger, start_iter=model.update)
    timer, start = AvgTimer(), time.time()
    amp = ("bf16" if cfg.get("bfloat16") else "fp16") if cfg.get("use_amp") else "off"
    det = "deterministic" if cfg.get("deterministic") else "non-deterministic"
    log.info(f"AMP: {amp}; {det}, seed {rt.seed}.")
    log.info(f"Starting at epoch {model.epoch}, update {model.update}.")

    prev_sigint = signal.getsignal(signal.SIGINT)
    stop = StopFlag()
    stop.install()
    try:
        prefetcher = CUDAPrefetcher(loader, rt.device)
        model.restore_rng()  # the last call before the loop
        epoch = model.epoch
        while True:
            model.epoch = epoch
            sampler.set_epoch(epoch)
            prefetcher.reset()  # iter(loader) under sigint_ignored() (T2 D19)
            while (batch := prefetcher.next()) is not None:
                model.feed_data(batch)
                if not model.optimize_parameters():  # inside a window
                    continue
                timer.record()
                u = model.update
                stopping = stop.requested(rt.device)  # all-reduced under DDP
                done = stopping or u >= total
                if u % print_freq == 0 or done:
                    # get_current_log is a collective under DDP: every rank calls it
                    log_vars = {"epoch": epoch, "iter": u, "lrs": model.current_lr()}
                    log_vars |= {
                        "time": timer.get_avg_time(),
                        **model.get_current_log(),
                    }
                    msg_logger(log_vars)
                if done:
                    if stopping:
                        log.info(f"stop requested: saving update {u}")
                    model.check_finite()  # raises before NaN reaches disk
                    model.save(final=True)  # state first, then exports + *_latest
                    if not stopping:
                        for vl in val_loaders:
                            model.validation(vl, u, tb_logger, save_img)
                    elapsed = datetime.timedelta(seconds=int(time.time() - start))
                    end = "Stopped" if stopping else "End of training"
                    log.info(f"{end} at update {u}. Time consumed: {elapsed}")
                    return
                val_due = bool(val_loaders) and bool(val_freq) and u % val_freq == 0
                save_due = u % save_freq == 0
                if val_due or save_due:
                    model.check_finite()  # all ranks
                if val_due:
                    for vl in val_loaders:
                        model.validation(vl, u, tb_logger, save_img)
                if save_due:
                    log.info(f"saving update {u}")
                    model.save()
                timer.start()
            epoch += 1
    finally:
        signal.signal(signal.SIGINT, prev_sigint)
        if tb_logger is not None:
            tb_logger.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = train_args(argv)  # argparse may SystemExit(2)
    get_root_logger()
    try:
        cfg = load_config(
            args.opt,
            is_train=True,
            root=ROOT,
            debug=args.debug,
            auto_resume=args.auto_resume,
        )
        validate_train(cfg)
        rt = setup_runtime(cfg, launcher=args.launcher)
        get_root_logger()  # rank known now: ranks != 0 log errors only
        prepare_run(cfg, sys.argv if argv is None else ["train.py", *argv])
        train(cfg, rt)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # second Ctrl-C: the last committed checkpoint stands
        log.error("interrupted; the last saved checkpoint stands")
        return 130
    except Exception:
        log.exception("training failed")
        return 1
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    sys.exit(main())
