"""Configuration loading and validation. Pure: no torch, argv, CUDA, RNG or filesystem writes."""

import difflib
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

# OTF degradation keys: valid in [degradations] and, after the merge, in [datasets.train]
_DEGRADATIONS: dict[str, Any] = {
    **{
        f"{k}{s}": t
        for s in ("", "2")
        for k, t in {
            "blur_kernel_size": int,
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
        "use_amp": bool,
        "bfloat16": bool,
        "fast_matmul": bool,
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
        "clamp": bool,
        "grad_clip": bool,
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
        "metrics": dict,
    },
    "logger": {
        "total_iter": int,
        "print_freq": int,
        "save_checkpoint_freq": int,
        "use_tb_logger": bool,
        "save_tb_img": bool,
        "keep_states": int,
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
        if phase.split("_")[0] not in PHASES:
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
    tile = cfg.get("val", {}).get("tile", 0)
    overlap = cfg.get("val", {}).get("tile_overlap", 16)
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
