"""Weight files, training states and RNG snapshots (design T2 §7, §8.1)."""

import contextlib
import os
import random
import tempfile
import time
from collections import OrderedDict
from collections.abc import Generator, Iterable, Mapping
from contextlib import contextmanager
from logging import Logger
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from neosr.config import ConfigError
from neosr.utils.rng import get_state as neosr_get_state
from neosr.utils.rng import set_state as neosr_set_state

STATE_FORMAT = 2
_PREFIXES = ("module.", "_orig_mod.")


def _strip(key: str) -> str:
    while key.startswith(_PREFIXES):
        key = key.split(".", 1)[1]
    return key


def _keep_metadata(src: Mapping, out: OrderedDict, rename=lambda k: k) -> OrderedDict:
    """Carry state_dict `_metadata` (per-module versions that _load_from_state_dict remaps read) over."""
    meta = getattr(src, "_metadata", None)
    if meta is not None:
        # a wrapper's root module path ("module", "_orig_mod") strips to the network root ""
        out._metadata = OrderedDict(  # type: ignore[attr-defined]
            (rename(k + "." if k else k).removesuffix("."), v) for k, v in meta.items()
        )
    return out


def strip_prefixes(sd: Mapping[str, Any]) -> OrderedDict:
    """Drop any number of leading DDP/compile prefixes and swa's `n_averaged`."""
    out: OrderedDict = OrderedDict()
    for key, v in sd.items():
        k = _strip(key)
        if k == "n_averaged":
            continue
        if k in out:
            msg = f"prefix stripping collides on {k!r}"
            raise KeyError(msg)
        out[k] = v
    return _keep_metadata(sd, out, _strip)


def cpu_copy(sd: Mapping[str, torch.Tensor]) -> OrderedDict:
    out = OrderedDict(
        (k, v.detach().to("cpu", memory_format=torch.contiguous_format, copy=True))
        for k, v in sd.items()
    )
    return _keep_metadata(sd, out)


def read_weights(path: str | Path, param_key: str | None = None) -> OrderedDict:
    # no mmap: legacy non-zip files reject it
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict):
        msg = f"{path}: expected a dict of weights, got {type(obj).__name__}"
        raise ConfigError(msg)
    if "format" in obj and "nets" in obj:
        msg = f"{path} is a training state; pretrain from models/net_*_<update>.pth or resume it"
        raise ConfigError(msg)
    if param_key is not None:
        if param_key not in obj:
            msg = f"{path}: param_key {param_key!r} not found; keys: {list(obj)[:10]}"
            raise ConfigError(msg)
        return strip_prefixes(obj[param_key])
    for key in ("params_ema", "params-ema", "params"):
        if key in obj:
            return strip_prefixes(obj[key])
    if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return strip_prefixes(obj)
    msg = f"{path}: no parameter mapping; keys: {list(obj)[:10]}"
    raise ConfigError(msg)


def load_weights(
    net: nn.Module, sd: Mapping[str, torch.Tensor], strict: bool, log: Logger
) -> None:
    """strict: exact names and shapes. Relaxed: drop shape mismatches, list missing/unexpected keys."""
    if not strict:
        own = net.state_dict()
        sd = _keep_metadata(sd, OrderedDict(sd))
        for k in [k for k in sd if k in own and own[k].shape != sd[k].shape]:
            log.warning(
                f"shape mismatch, skipped {k}: {tuple(sd[k].shape)} -> {tuple(own[k].shape)}"
            )
            del sd[k]
        if not sd.keys() & own.keys():
            msg = "pretrained file matches no tensor of the network"
            raise ConfigError(msg)
    res = net.load_state_dict(sd, strict=strict)
    for k in res.missing_keys:
        log.warning(f"missing: {k}")
    for k in res.unexpected_keys:
        log.warning(f"unexpected: {k}")


def _umask() -> int:
    mask = os.umask(0)  # the only portable way to read it
    os.umask(mask)
    return mask


def atomic_save(obj: Any, path: str | Path, attempts: int = 3) -> None:
    """Write to a sibling temp file, fsync, then rename over `path`: readers never see a partial file."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        # mkstemp makes 0600; torch.save used to honour the umask
        Path(tmp).chmod(0o666 & ~_umask())
        with os.fdopen(fd, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        for _ in range(attempts - 1):  # Windows: a viewer may hold the target open
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                time.sleep(1)
        else:  # last (or only) attempt: always runs, its error propagates
            os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    if os.name == "posix":
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)


def _numbered(d: Path) -> list[Path]:
    return sorted(
        (p for p in Path(d).glob("*.state") if p.stem.isdigit() and p.is_file()),
        key=lambda p: int(p.stem),
    )


def latest_state(d: str | Path) -> Path | None:
    s = _numbered(Path(d))
    return s[-1] if s else None


def prune_states(d: str | Path, keep: int) -> None:
    """Keep the newest `keep` numbered states; keep <= 0 keeps all."""
    for p in _numbered(Path(d))[:-keep] if keep > 0 else []:
        p.unlink(missing_ok=True)


def prune_exports(d: str | Path, keep: int, names: Iterable[str]) -> None:
    """Keep the newest `keep` numbered exports (<name>_<update>.pth) per net; keep <= 0 keeps all.
    `*_latest` and any other file are never touched."""
    if keep <= 0:
        return
    for name in names:
        numbered = sorted(
            (int(u), f)
            for f in Path(d).glob(f"{name}_*.pth")
            if (u := f.stem.removeprefix(f"{name}_")).isdigit() and f.is_file()
        )
        for _, f in numbered[:-keep]:
            f.unlink(missing_ok=True)


def rng_state(
    device: torch.device | None = None, loader_gen: torch.Generator | None = None
) -> dict:
    """Snapshot of every process-wide stream; loads with torch.load(weights_only=True)."""
    _, key, pos, has_gauss, gauss = np.random.get_state()
    cuda = device is not None and device.type == "cuda"
    return {
        "python": random.getstate(),
        "numpy": {
            "key": torch.from_numpy(key.astype(np.int64)),
            "pos": int(pos),
            "has_gauss": int(has_gauss),
            "gauss": float(gauss),
        },
        "neosr": neosr_get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if cuda else None,
        "loader": loader_gen.get_state() if loader_gen is not None else None,
    }


def set_rng_state(
    s: dict,
    device: torch.device | None = None,
    loader_gen: torch.Generator | None = None,
) -> None:
    random.setstate(s["python"])
    n = s["numpy"]
    np.random.set_state((
        "MT19937",
        n["key"].numpy().astype(np.uint32),
        n["pos"],
        n["has_gauss"],
        n["gauss"],
    ))
    if s["neosr"] is not None:  # None in states saved before the stream was captured
        neosr_set_state(s["neosr"])
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None and device is not None and device.type == "cuda":
        torch.cuda.set_rng_state(s["cuda"], device)
    if s["loader"] is not None and loader_gen is not None:
        loader_gen.set_state(s["loader"])


@contextmanager
def preserved_rng(device: torch.device | None = None) -> Generator[None, None, None]:
    """Run a block (validation, export) without disturbing the training streams."""
    s = rng_state(device)
    try:
        yield
    finally:
        set_rng_state(s, device)
