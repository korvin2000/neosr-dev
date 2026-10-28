"""Configuration loading and validation. Pure: no torch, argv, CUDA, RNG or filesystem writes."""

import difflib
import math
import tomllib
import warnings
from collections.abc import Iterable
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Invalid configuration: wrong/missing key, unknown type or removed feature (exit code 2)."""

    def __init__(
        self, msg: str, *, key: str | None = None, known: Iterable[str] = ()
    ) -> None:
        super().__init__(msg + _did_you_mean(key, known))


class ConfigWarning(UserWarning):
    """Suspicious but accepted configuration, e.g. an unknown key."""


def _did_you_mean(key: str | None, known: Iterable[str]) -> str:
    if key and (m := difflib.get_close_matches(key, list(known), n=1)):
        return f" (did you mean {m[0]!r}?)"
    return ""


REQUIRED = ("name", "model_type", "scale", "network_g", "datasets")
SCALES = frozenset({1, 2, 3, 4, 8})
PHASES = frozenset({"train", "val", "test"})
# val.metrics types where a lower value is better: the default of their `better` (M-14)
METRIC_LOWER = frozenset({"calculate_dists"})

# OTF degradation keys: valid in [degradations] and, after the merge, in [datasets.train]
_DEGRADATIONS: dict[str, Any] = {
    **{
        f"{k}{s}": t
        for s in ("", "2")
        for k, t in {
            "kernel_list": list,
            "kernel_prob": list,
            "blur_sigma": list,
            "blur_sigmay": list,
            "betag_range": list,
            "betap_range": list,
            "sinc_prob": float,
            "resize_prob": list,
            "resize_range": list,
            "gaussian_noise_prob": float,
            "noise_range": list,
            "poisson_scale_range": list,
            "gray_noise_prob": float,
            "jpeg_range": list,
        }.items()
    },
    "rotation": list,
    "final_sinc_prob": float,
    "second_blur_prob": float,
    "jpeg_compress": bool,
}

# Reviewed inventory: dotted section -> {key: type}; "*" matches one level. `float` also accepts int,
# no type accepts bool except bool. Sections listed in OPAQUE are validated by their constructors.
SCHEMA: dict[str, dict[str, Any]] = {
    "": {
        "name": str,
        "model_type": str,
        "scale": int,
        "num_gpu": (int, str),
        "manual_seed": int,
        "deterministic": bool,
        "use_amp": bool,
        "bfloat16": bool,
        "fast_matmul": bool,
        "fp16_accumulation": bool,
        "compile": bool,
        "compile_mode": str,
        "find_unused_parameters": bool,
        "input_size": int,
        "in_chans": int,
        "print_network": bool,
        "network_g": dict,
        "network_d": dict,
        "datasets": dict,
        "train": dict,
        "val": dict,
        "logger": dict,
        "path": dict,
        "degradations": dict,
        "dist_params": dict,
    },
    "datasets.*": {
        "type": str,
        "name": str,
        "dataroot_gt": str,
        "dataroot_lq": str,
        "meta_info": str,
        "meta_info_file": str,
        "patch_size": int,
        "batch_size": int,
        "accumulate": int,
        "num_worker_per_gpu": (int, str),
        "pin_memory": bool,
        "persistent_workers": bool,
        "dataset_enlarge_ratio": int,
        "augmentation": list,
        "aug_prob": list,
        "aug_multi_prob": float,
        "aug_antialias": bool,
        "tight_canvas": bool,
        "use_hflip": bool,
        "use_rot": bool,
        "color": str,
        "mean": list,
        "std": list,
        "queue_size": int,
        **_DEGRADATIONS,
    },
    "degradations": _DEGRADATIONS,
    "train": {
        "optim_g": dict,
        "optim_d": dict,
        "scheduler": dict,
        "ema": float,
        "ema_power": float,
        "clamp": (bool, str),
        "grad_clip": (bool, float, str),
        "match_lq_colors": bool,
        "wavelet_guided": bool,
        "wavelet_init": int,
        "max_skip_streak": int,
        "enable_qat": bool,
        "qat_type": str,
        "fused_init": bool,
        "quantize_residual": bool,
        "calibration_iters": int,
        "qat_freeze_norm_iter": int,
        "qat_freeze_observers_iter": int,
    },
    "val": {
        "val_freq": int,
        "save_img": bool,
        "tile": int,
        "tile_overlap": int,
        "pbar": bool,
        "suffix": str,
        "save_lq": bool,
        "save_best": str,
        "metrics": dict,
    },
    "logger": {
        "total_iter": int,
        "print_freq": int,
        "save_checkpoint_freq": int,
        "use_tb_logger": bool,
        "save_tb_img": bool,
        "keep_states": int,
        "keep_exports": int,
        "wandb": dict,
    },
    "path": {
        "pretrain_network_g": str,
        "pretrain_network_d": str,
        "param_key_g": str,
        "param_key_d": str,
        "strict_load_g": bool,
        "strict_load_d": bool,
        "resume_state": str,
        "experiments_root": str,
        "results_root": str,
        "print_network": bool,
    },
    "dist_params": {"backend": str, "port": int},
}
OPAQUE = (
    "network_g",
    "network_d",
    "train.*_opt",
    "train.optim_g",
    "train.optim_d",
    "train.scheduler",
    "val.metrics.*",
    "logger.wandb",
)
REMOVED: dict[str, str] = {
    "train.sam": "SAM/FSAM was removed",
    "train.sam_init": "SAM/FSAM was removed",
    "train.eco": "ECO was removed",
    "train.eco_init": "ECO was removed",
    "train.eco_iters": "ECO was removed",
    "train.eco_schedule": "ECO was removed",
    "train.warmup_iter": "warmup_iter was removed; use the optimizer's warmup_steps",
    "train.gw_opt": "gw_loss was removed",
    "train.color_opt": "color_loss was removed",
    "train.luma_opt": "luma_loss was removed",
    "path.ignore_resume_networks": "ignore_resume_networks was removed; use pretrain_network_* for a new run",
}


def load_config(
    path: str | Path,
    *,
    is_train: bool,
    root: str | Path,
    debug: bool = False,
    auto_resume: bool = False,
) -> dict[str, Any]:
    """Read, validate and derive a TOML config. Raises ConfigError; unknown keys warn ConfigWarning.

    Absent optional keys and sections are never filled with defaults: readers keep their `.get`.
    """
    path = Path(path)
    if path.suffix != ".toml":
        msg = f"{path}: expected a .toml file (neosr uses TOML configs, see options/)"
        raise ConfigError(msg)
    try:
        cfg = tomllib.loads(path.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        msg = f"{path}: {e} (on Windows, quote paths with single quotes)"
        raise ConfigError(msg) from e
    for key in REQUIRED:
        if key not in cfg:
            msg = f"missing required key {key!r}"
            raise ConfigError(msg)
    _check(cfg, "")
    for phase in cfg["datasets"]:
        # train has no suffix: loaders read the literal [datasets.train]
        if phase != "train" and phase.split("_")[0] not in PHASES - {"train"}:
            msg = f"[datasets.{phase}] unknown phase; use train, val[_N] or test[_N]"
            raise ConfigError(msg)
    _derive(
        cfg,
        path=path,
        root=Path(root),
        is_train=is_train,
        debug=debug,
        auto_resume=auto_resume,
    )
    return cfg


def _section_schema(section: str) -> dict[str, Any] | None:
    if section in SCHEMA:
        return SCHEMA[section]
    return next(
        (v for k, v in SCHEMA.items() if "*" in k and fnmatchcase(section, k)), None
    )


def _type_ok(val: Any, expected: Any) -> bool:
    expected = expected if isinstance(expected, tuple) else (expected,)
    if isinstance(val, bool):
        return bool in expected
    return isinstance(val, expected) or (float in expected and isinstance(val, int))


def _type_name(expected: Any) -> str:
    expected = expected if isinstance(expected, tuple) else (expected,)
    return " or ".join(
        {dict: "table", list: "array"}.get(t, t.__name__) for t in expected
    )


def _check(node: dict[str, Any], section: str) -> None:
    schema = _section_schema(section)
    for key, val in node.items():
        dotted = f"{section}.{key}" if section else key
        if dotted in REMOVED:
            msg = f"[{section or 'root'}] {key}: {REMOVED[dotted]}"
            raise ConfigError(msg)
        if any(fnmatchcase(dotted, p) for p in OPAQUE):
            if not isinstance(val, dict):
                msg = f"{dotted} must be a table, got {type(val).__name__}"
                raise ConfigError(msg)
            continue
        # phase names are free-form (train, val, val_1, test_1, ...) and checked by load_config
        if section == "datasets":
            expected: Any = dict
        elif schema is None or (expected := schema.get(key)) is None:
            known = schema or ()
            msg = f"unknown key {dotted!r} is ignored{_did_you_mean(key, known)}"
            warnings.warn(msg, ConfigWarning, stacklevel=4)
            continue
        if not _type_ok(val, expected):
            msg = f"{dotted} must be {_type_name(expected)}, got {type(val).__name__} {val!r}"
            raise ConfigError(msg)
        if isinstance(val, dict):
            _check(val, dotted)


_PATH_KEYS = {
    "pretrain_network_g",
    "pretrain_network_d",
    "resume_state",
    "experiments_root",
    "results_root",
}


def _expand(val: str) -> str:
    return str(Path(val).expanduser())


def _derive(
    cfg: dict[str, Any],
    *,
    path: Path,
    root: Path,
    is_train: bool,
    debug: bool,
    auto_resume: bool,
) -> None:
    cfg.update(
        root_path=str(root),
        opt_path=str(path),
        is_train=is_train,
        auto_resume=auto_resume,
    )
    if debug and not cfg["name"].startswith("debug"):
        cfg["name"] = "debug_" + cfg["name"]

    for phase, ds in cfg["datasets"].items():
        ds["phase"] = phase.split("_")[0]
        ds["scale"] = cfg["scale"]
        if ds["phase"] != "train":  # results, best metrics and TB tags are keyed by it
            ds.setdefault("name", phase)
        for key in ("dataroot_gt", "dataroot_lq"):
            if key in ds:
                ds[key] = _expand(ds[key])

    train_ds = cfg["datasets"].get("train")
    if train_ds is not None and "degradations" in cfg:
        for key, val in cfg["degradations"].items():
            if (
                key not in _DEGRADATIONS
            ):  # already warned as unknown: must not overwrite train keys
                continue
            if key in train_ds and train_ds[key] != val:
                warnings.warn(
                    f"{key!r} is set in [datasets.train] and [degradations]; "
                    f"[degradations] wins ({val!r})",
                    ConfigWarning,
                    stacklevel=3,
                )
            train_ds[key] = val

    p = cfg.setdefault("path", {})
    for key in _PATH_KEYS & p.keys():
        p[key] = _expand(p[key])
    name = cfg["name"]
    if is_train:
        exp = Path(p.get("experiments_root", root / "experiments")) / name
        p.update(
            experiments_root=exp,
            models=exp / "models",
            training_states=exp / "training_states",
            log=exp,
            visualization=exp / "visualization",
        )
        if "debug" in name:
            if "val" in cfg:
                cfg["val"]["val_freq"] = 8
            cfg.setdefault("logger", {}).update(print_freq=1, save_checkpoint_freq=8)
    else:
        res = Path(p.get("results_root", root / "experiments" / "results")) / name
        p.update(results_root=res, log=res, visualization=res)


def _plain(val: Any) -> bool:
    """TOML strings, numbers, booleans, arrays and tables: what an export's metadata may hold
    (torch.load(weights_only=True) rejects dates and times)."""
    if isinstance(val, dict):
        return all(_plain(v) for v in val.values())
    if isinstance(val, list):
        return all(_plain(v) for v in val)
    return isinstance(val, str | int | float | bool)


def validate_train(cfg: dict[str, Any]) -> None:
    """Structural cross-key rules of a training config; raises ConfigError naming the dotted key."""
    if cfg["scale"] not in SCALES:
        msg = f"scale={cfg['scale']} is not supported; use one of {sorted(SCALES)}"
        raise ConfigError(msg)
    if "train" not in cfg["datasets"]:
        msg = "training needs a [datasets.train] section"
        raise ConfigError(msg)
    if bad := [k for k in cfg["datasets"] if k.split("_")[0] == "test"]:
        msg = (
            f"[datasets.{bad[0]}] test datasets belong in an inference config; use val"
        )
        raise ConfigError(msg)
    for key in ("network_g", "network_d"):  # copied into every export (stage 3 A1)
        if not _plain(cfg.get(key) or {}):
            msg = f"[{key}] may hold only strings, numbers, booleans, arrays and tables"
            raise ConfigError(msg)
    train = cfg.get("train")
    if not train or "optim_g" not in train:
        msg = "training needs a [train.optim_g] section"
        raise ConfigError(msg)
    if "network_d" in cfg and "optim_d" not in train:
        msg = "network_d is set but train.optim_d is not: add an optimizer or remove network_d"
        raise ConfigError(msg)
    if "optim_d" in train and "network_d" not in cfg:
        msg = "train.optim_d is set but network_d is not: add a discriminator or remove optim_d"
        raise ConfigError(msg)
    if train.get("max_skip_streak", 1) < 1:
        msg = "train.max_skip_streak must be >= 1"
        raise ConfigError(msg)
    if not 0 <= (ema := train.get("ema", 0)) < 1:  # also rejects nan and inf
        msg = f"train.ema must be in [0, 1) (0 = off), got {ema}"
        raise ConfigError(msg)
    if (power := train.get("ema_power")) is not None and not power > 0:
        msg = f"train.ema_power must be > 0, got {power}"
        raise ConfigError(msg)
    if isinstance(clamp := train.get("clamp", "restore"), str) and clamp != "restore":
        msg = f"train.clamp must be true, false or 'restore', got {clamp!r}"
        raise ConfigError(msg)
    clip = train.get("grad_clip", True)
    if (
        clip != "spike"
        if isinstance(clip, str)
        else not isinstance(clip, bool) and not (clip > 0 and math.isfinite(clip))
    ):
        msg = f"train.grad_clip must be true, false, a max-norm > 0 or 'spike', got {clip!r}"
        raise ConfigError(msg)
    val = cfg.get("val") or {}
    if (val_freq := val.get("val_freq")) is not None and val_freq < 1:
        msg = f"val.val_freq must be >= 1 (remove it to disable validation), got {val_freq}"
        raise ConfigError(msg)
    for name, metric in (val.get("metrics") or {}).items():
        if metric.get("better", "higher") not in {"higher", "lower"}:
            msg = f"val.metrics.{name}.better must be 'higher' or 'lower', got {metric['better']!r}"
            raise ConfigError(msg)
    if (best := val.get("save_best")) is not None and best not in (
        metrics := val.get("metrics") or {}
    ):
        msg = f"val.save_best = {best!r} is not a key of val.metrics"
        raise ConfigError(msg, key=best, known=metrics)
    tile, overlap = val.get("tile", 0), val.get("tile_overlap", 16)
    if tile > 0 and not tile > 2 * overlap >= 0:
        msg = f"val.tile={tile} with val.tile_overlap={overlap}: need tile > 2 * tile_overlap >= 0"
        raise ConfigError(msg)
    if cfg.get("bfloat16") and not cfg.get("use_amp"):
        msg = (
            "bfloat16 has no effect without use_amp: enable use_amp or remove bfloat16"
        )
        raise ConfigError(msg)
    ds = cfg["datasets"]["train"]
    for key in ("patch_size", "batch_size"):
        if key not in ds:
            msg = f"datasets.train.{key} is required"
            raise ConfigError(msg)
        if ds[key] < 1:
            msg = f"datasets.train.{key} must be positive, got {ds[key]}"
            raise ConfigError(msg)
    if ds.get("accumulate", 1) < 1:
        msg = f"datasets.train.accumulate must be >= 1, got {ds['accumulate']}"
        raise ConfigError(msg)
    if ds.get("persistent_workers") and ds.get("num_worker_per_gpu") == 0:
        msg = "datasets.train.persistent_workers needs num_worker_per_gpu > 0"
        raise ConfigError(msg)
    if not 0 <= (multi := ds.get("aug_multi_prob", 0.3)) <= 1:
        msg = f"datasets.train.aug_multi_prob must be in [0, 1], got {multi}"
        raise ConfigError(msg)
    names: dict[str, str] = {}
    for phase, d in cfg["datasets"].items():
        workers = d.get("num_worker_per_gpu", 4)
        if workers != "auto" and (isinstance(workers, str) or workers < 0):
            msg = f"datasets.{phase}.num_worker_per_gpu must be an integer >= 0 or 'auto', got {workers!r}"
            raise ConfigError(msg)
        if d.get("dataset_enlarge_ratio", 1) < 1:
            msg = f"datasets.{phase}.dataset_enlarge_ratio must be >= 1, got {d['dataset_enlarge_ratio']}"
            raise ConfigError(msg)
        # before the model build; the datasets reject them too
        for key in ("mean", "std"):
            if d.get(key) is not None:
                msg = f"[datasets.{phase}] {key!r} is not supported (images stay in [0, 1]); remove it"
                raise ConfigError(msg, key=key)
        if phase != "train":
            if (name := d.get("name", phase)) in names:
                msg = (
                    f"datasets.{phase}: name {name!r} is also datasets.{names[name]}'s"
                )
                raise ConfigError(msg)
            names[name] = phase
    lg = cfg.get("logger") or {}
    for key in ("total_iter", "print_freq", "save_checkpoint_freq"):
        n = lg.get(key)
        if (n is None and key == "save_checkpoint_freq") or (n is not None and n < 1):
            msg = f"logger.{key} must be set to a positive number of updates"
            raise ConfigError(msg)
    if train.get("enable_qat"):
        _validate_qat(cfg, train)


def qat_schedule(train: dict[str, Any], total_iter: int) -> tuple[int, int, int]:
    """QAT phase boundaries in updates (design T5 D12/D13): (calibration_iters, qat_freeze_norm_iter,
    qat_freeze_observers_iter). Updates <= calibration only observe; freezes start at their update;
    -1 = never."""
    return (
        train.get("calibration_iters", 0),
        train.get("qat_freeze_norm_iter", int(0.7 * total_iter)),
        train.get("qat_freeze_observers_iter", int(0.8 * total_iter)),
    )


def _validate_qat(cfg: dict[str, Any], train: dict[str, Any]) -> None:
    if (kind := train.get("qat_type", "int8")) != "int8":
        msg = f"train.qat_type={kind!r} is not supported; use 'int8' (symmetric QDQ for TensorRT/ONNX Runtime)"
        raise ConfigError(msg)
    if not train.get("fused_init", True):
        msg = "train.fused_init=false (QAT on unfused reparam branches) is not supported; remove it"
        raise ConfigError(msg)
    net = cfg["network_g"]
    t_res, n_res = train.get("quantize_residual"), net.get("quantize_residual")
    if t_res is not None and n_res is not None and t_res != n_res:
        msg = f"train.quantize_residual={t_res} contradicts network_g.quantize_residual={n_res}"
        raise ConfigError(msg)
    p = cfg.get("path") or {}
    if net.get("fused_init") and not (
        p.get("pretrain_network_g") or p.get("resume_state") or cfg.get("auto_resume")
    ):
        msg = (
            "network_g.fused_init=true builds the deploy topology (folded norms); QAT from scratch "
            "needs fused_init=false, or set path.pretrain_network_g"
        )
        raise ConfigError(msg)
    total = (cfg.get("logger") or {}).get("total_iter", 1000000)
    calib, norm_at, obs_at = qat_schedule(train, total)
    if calib < 0:
        msg = f"train.calibration_iters must be >= 0, got {calib}"
        raise ConfigError(msg)
    # defaults: 0.7 / 0.8 * logger.total_iter
    for key, at, low in (
        ("qat_freeze_norm_iter", norm_at, calib),
        ("qat_freeze_observers_iter", obs_at, max(calib, 1)),
    ):
        if at != -1 and at <= low:
            msg = (
                f"train.{key}={at} must be -1 (never) or > {low}: train.calibration_iters={calib} "
                "and at least one observed update come first"
            )
            raise ConfigError(msg)
    if -1 not in (norm_at, obs_at) and norm_at > obs_at:
        msg = f"train.qat_freeze_norm_iter={norm_at} must be <= train.qat_freeze_observers_iter={obs_at}"
        raise ConfigError(msg)
