"""Generic trainer plumbing (design T2 §2, §6.2, §8.3): runtime and device, execution wrappers, the
validation driver, save / training state / resume / RNG restore, best metrics and state pruning."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, MultiStepLR

from neosr.config import METRIC_LOWER, ConfigError
from neosr.losses.gan_loss import batch_coupled
from neosr.metrics import build_metric
from neosr.models.checkpoint import (
    STATE_FORMAT,
    atomic_save,
    preserved_rng,
    prune_exports,
    prune_states,
    rng_state,
    set_rng_state,
)
from neosr.optimizers import adamw_sf, adan, adan_sf
from neosr.runtime import Runtime
from neosr.utils import get_root_logger, tc
from neosr.utils.factory import construct

# T4 §3.10; looked up lower-cased, so "AdamW" and "adamw" both work
OPTIMIZERS: dict[str, type[Optimizer]] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "nadam": torch.optim.NAdam,
    "adan": adan,
    "adamw_sf": adamw_sf,
    "adan_sf": adan_sf,
}
SCHEDULERS: dict[str, type[LRScheduler]] = {
    "multisteplr": MultiStepLR,
    "cosineannealing": CosineAnnealingLR,
}


def _lookup(
    table: Mapping[str, Any], cfg: Mapping[str, Any], section: str
) -> tuple[str, Any, dict]:
    kwargs = dict(cfg)
    kind = str(kwargs.pop("type", None)).lower()
    if kind not in table:
        msg = f"[{section}] unknown type {cfg.get('type')!r}"
        raise ConfigError(msg, key=kind, known=table)
    return kind, table[kind], kwargs


def build_optimizer(
    cfg: Mapping[str, Any], params: list[nn.Parameter], section: str
) -> tuple[str, Optimizer]:
    """(table key, optimizer); the key is what a training state records as the optimizer type."""
    kind, cls, kwargs = _lookup(OPTIMIZERS, cfg, section)
    return kind, construct(cls, section, kind, kwargs, params=params)


def build_schedulers(
    cfg: Mapping[str, Any] | None, opt: Optimizer
) -> list[LRScheduler]:
    if not cfg:
        return []
    kind, cls, kwargs = _lookup(SCHEDULERS, cfg, "train.scheduler")
    return [construct(cls, "train.scheduler", kind, kwargs, optimizer=opt)]


def _flat(d: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, Mapping):
            out |= {f"{k}.{kk}": vv for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def dict_diff(
    old: Mapping[str, Any], new: Mapping[str, Any]
) -> dict[str, tuple[Any, Any]]:
    """{dotted key: (old, new)} for differing keys; nested dicts are flattened one level."""
    a, b = _flat(old), _flat(new)
    return {
        k: (a.get(k), b.get(k))
        for k in sorted(a.keys() | b.keys())
        if a.get(k) != b.get(k)
    }


class base:
    """Trainer plumbing shared by every model type. Subclasses build the nets and phases."""

    def __init__(self, opt: dict[str, Any], runtime: Runtime) -> None:
        self.opt, self.rt, self.device = opt, runtime, runtime.device
        self.is_train: bool = opt["is_train"]
        self.scale: int = opt["scale"]
        # T2 §3 step 0: everything later steps may leave unset
        self.net_g: nn.Module
        self.net_d: nn.Module | None = None
        self.net_g_fwd = self.net_d_fwd = self.net_d_adv = None
        self.ema = None
        self.nets: dict[str, nn.Module] = {}
        self.phases: dict[str, Any] = {}
        self.update = self.epoch = self.micro = 0
        self.accumulate = 1
        self.best_metric_results: dict[str, dict[str, dict[str, Any]]] = {}
        self._evaluating = False
        self._pending_rng: dict | None = None
        self.data_generator: torch.Generator | None = None
        val = opt.get("val") or {}
        self.tile, self.tile_overlap = val.get("tile", -1), val.get("tile_overlap", 16)
        lg = opt.get("logger") or {}
        self.keep_states, self.keep_exports = (
            lg.get("keep_states", 3),
            lg.get("keep_exports", 0),
        )
        self.saved_update: int | None = None  # the update of this process's last save
        self.metrics = {
            k: build_metric(v, self.device, section=f"val.metrics.{k}")
            for k, v in (val.get("metrics") or {}).items()
        }

    # implemented by the SR trainer (image.py)
    def validate(self, loader, current_iter, tb_logger, save_img: bool) -> None:
        raise NotImplementedError

    def export_state_dicts(self) -> dict[str, dict[str, torch.Tensor]]:
        raise NotImplementedError

    def run_record(self) -> dict[str, Any]:
        raise NotImplementedError

    # ---------------------------------------------------------------- execution wrappers
    def wrap(
        self, net: nn.Module | None, *, broadcast_buffers: bool = True, ddp: bool = True
    ) -> nn.Module | None:
        """Training-forward wrapper: DDP (multi-process, unless `ddp=False`), then torch.compile
        (T4 P7). D is compiled when its update is one forward (gan_loss.batch_coupled is False):
        a compiled spectral-norm D run on real and fake before one backward fails the saved-tensor
        version check (perf.md). Never used to load, save, average or validate: those use the bare
        module."""
        if net is None:
            return None
        compile_it = bool(self.opt.get("compile")) and (
            net is self.net_g or not batch_coupled(net)
        )
        if ddp and self.rt.distributed:
            if not broadcast_buffers:
                # the constructor then syncs parameters only: start the buffers (SN u/v, seeded
                # per rank) from rank 0 once; from synced weights they then evolve identically
                for b in net.buffers():
                    torch.distributed.broadcast(b, src=0)
            net = DistributedDataParallel(
                net,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                broadcast_buffers=broadcast_buffers,
                find_unused_parameters=self.opt.get("find_unused_parameters", False),
                gradient_as_bucket_view=True,  # grads are bucket views: no extra gradient copy
            )
        if compile_it:
            net = torch.compile(
                net, mode=self.opt.get("compile_mode", "default"), dynamic=False
            )
        return net  # type: ignore[return-value]

    def log_network(self, net: nn.Module, label: str) -> None:
        if self.rt.rank != 0:
            return
        n = sum(p.numel() for p in net.parameters())
        log = get_root_logger()
        log.info(
            f"[{type(self).__name__}] {label}: {type(net).__name__}, parameters: {n:,d}"
        )
        if self.opt.get("print_network") or self.opt["path"].get("print_network"):
            log.info(str(net))

    # ---------------------------------------------------------------- validation
    def validation(
        self, dataloader, current_iter, tb_logger, save_img: bool = True
    ) -> None:
        """Rank 0 validates after a barrier; every training RNG stream is left untouched. A rank-0
        failure is all-reduced, so every rank raises instead of waiting at a barrier (F23)."""
        dist = torch.distributed.is_initialized()
        if dist:
            torch.distributed.barrier()
        err: Exception | None = None
        if self.rt.rank == 0:
            from neosr.metrics.calculate import release_metric_nets  # noqa: PLC0415

            try:
                with preserved_rng(self.device):
                    self.validate(dataloader, current_iter, tb_logger, save_img)
            except Exception as e:  # noqa: BLE001 - re-raised below, after the collective
                err = e
            finally:  # DISTS/TOPIQ weights live for one validation pass only
                release_metric_nets()
        if dist:  # also the closing barrier
            failed = torch.tensor(float(err is not None), device=self.device)
            torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
            if err is None and bool(failed):
                msg = "validation failed on rank 0"
                raise RuntimeError(msg)
        if err is not None:
            raise err

    def update_best(self, dataset: str, metric: str, val: float, current_iter) -> None:
        cfg = self.opt["val"]["metrics"][metric]
        better = cfg.get("better") or (
            "lower" if cfg.get("type") in METRIC_LOWER else "higher"
        )
        init = float("-inf") if better == "higher" else float("inf")
        recs = self.best_metric_results.setdefault(dataset, {})
        rec = recs.get(metric)
        if rec is not None and rec.get("better") != better:
            # a resumed record kept the other direction (DISTS was "higher" before M-14)
            get_root_logger().warning(
                f"val.metrics.{metric} ({dataset}): better = {better!r} now; the saved best "
                f"{rec['val']:.4f} @ {rec['iter']} ({rec.get('better')!r}) starts over"
            )
            rec = None
        if rec is None:
            rec = recs[metric] = {"better": better, "val": init, "iter": -1}
        if (val >= rec["val"]) if rec["better"] == "higher" else (val <= rec["val"]):
            rec["val"], rec["iter"] = float(val), current_iter

    def log_metrics(
        self, dataset: str, results: Mapping[str, float], current_iter, tb_logger
    ) -> None:
        msg = f"Validation {dataset}\n\n"
        for k, v in results.items():
            best = self.best_metric_results[dataset][k]
            msg += (
                f"\t # {k}: {v:.4f}{tc.light_green}........ Best: {best['val']:.4f} @ "
                f"{best['iter']} iter{tc.end}\n"
            )
        get_root_logger().info(msg)
        if tb_logger:
            for k, v in results.items():
                tb_logger.add_scalar(f"metrics/{dataset}/{k}", v, current_iter)

    # ---------------------------------------------------------------- checkpoints
    def save(self, *, final: bool = False) -> None:
        """State first (authoritative), then the exports (and `*_latest` when final), then pruning."""
        if self._evaluating or self.micro % self.accumulate:
            msg = "save() only at an update boundary, outside eval_weights()"
            raise RuntimeError(msg)
        first, self.saved_update = self.saved_update is None, self.update
        if self.rt.rank != 0:
            return
        u, p = self.update, self.opt["path"]
        states, models = Path(p["training_states"]), Path(p["models"])
        states.mkdir(parents=True, exist_ok=True)
        models.mkdir(parents=True, exist_ok=True)
        atomic_save(self.training_state(), states / f"{u}.state")
        exports = self.export_state_dicts()
        for name, sd in exports.items():
            payload = {"params": sd}
            atomic_save(payload, models / f"{name}_{u}.pth")
            if final:
                atomic_save(payload, models / f"{name}_latest.pth")
        prune_states(states, self.keep_states)
        prune_exports(models, self.keep_exports, exports)
        if first:
            mib = sum((models / f"{n}_{u}.pth").stat().st_size for n in exports) / 2**20
            get_root_logger().info(
                f"one export set is {mib:.1f} MiB; logger.keep_exports = {self.keep_exports} "
                "(0 keeps every set)"
            )

    def training_state(self) -> dict[str, Any]:
        """Format 2 (T2 §7.2): self-contained, loads with torch.load(weights_only=True)."""
        return {
            "format": STATE_FORMAT,
            "update": self.update,
            "epoch": self.epoch,
            "nets": {n: m.state_dict() for n, m in self.nets.items()},
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "phases": {n: ph.state_dict() for n, ph in self.phases.items()},
            "rng": rng_state(self.device, self.data_generator),
            "best": self.best_metric_results,
            "run": self.run_record(),
        }

    def resume(self, path: Path) -> None:
        """Load a format-2 state into the bare modules and phases (before wrapping). An incompatible
        configuration raises ConfigError (T2 §7.3); there is no partial or fallback resume."""
        log = get_root_logger()
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e:
            msg = f"cannot read training state {path}: {e}"
            raise RuntimeError(msg) from e
        if "format" not in state:
            msg = (
                f"{path}: legacy training state cannot be resumed; start a new run with "
                f"pretrain_network_g = models/net_g_{state.get('iter')}.pth (and _d)"
            )
            raise ConfigError(msg)
        if state["format"] != STATE_FORMAT:
            msg = f"{path}: unsupported state format {state['format']}"
            raise ConfigError(msg)
        for kind in ("nets", "phases"):
            saved, now = sorted(state[kind]), sorted(getattr(self, kind))
            if saved != now:
                msg = f"{path}: {kind} {saved} saved, {now} configured; use pretrain_network_* for a new run"
                raise ConfigError(msg)
        for n, net in self.nets.items():
            try:
                net.load_state_dict(state["nets"][n])  # strict
            except RuntimeError as e:
                msg = f"{path}: {n} does not match the configured network:\n{e}"
                raise ConfigError(msg) from e
        for n, ph in self.phases.items():
            ph.load_state_dict(state["phases"][n], log)
        if self.ema is not None and state["ema"] is not None:
            self.ema.load_state_dict(state["ema"])
        elif self.ema is not None:
            # still uninitialised: the first update copies live G; until then evaluate the resumed weights
            self.ema.module.load_state_dict(self.net_g.state_dict())
            log.info(
                "ema enabled since the last save: the first update copies the live generator"
            )
        elif state["ema"] is not None:
            log.warning("saved EMA ignored: ema is disabled")
        saved = _flat(state["run"])
        for key, (old, new) in dict_diff(state["run"], self.run_record()).items():
            # a key absent from the saved record was recorded only since that state; train/data
            # hold the record's own keys, the other sections config sub-keys (a new one is a change)
            sec = key.split(".")[0]
            if key in saved or (sec in state["run"] and sec not in ("train", "data")):
                log.warning(f"changed since {path.name}: {key}: {old!r} -> {new!r}")
        self.update, self.epoch = state["update"], state["epoch"]
        self.best_metric_results = state["best"]
        self._pending_rng = state["rng"]
        log.info(
            f"resumed {path.name}: update {self.update}, epoch {self.epoch}. Exact: weights, EMA, "
            "optimizers, schedulers, scalers, counters, model-side RNG (rank 0). "
            "Restarted: data order, augmentation, OTF pool."
        )

    def restore_rng(self) -> None:
        """train.py: the last call before the loop, after the loader exists."""
        if self._pending_rng is not None and self.rt.rank == 0:
            set_rng_state(self._pending_rng, self.device, self.data_generator)
        self._pending_rng = None
