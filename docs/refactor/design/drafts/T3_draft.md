# T3 — Configuration, construction, imports, RNG, devices, error handling (merged design)

Inputs merged: Opus position, Fable proposal (kept verbatim as `design/T3_fable_proposal.md`, with probes/benchmarks in the same dir), Codex proposal. Conflicts were settled by reading `/home/korvin/apps/neosr-dev` (read-only); each settlement cites the code. Repo untouched.

## 0. The design in eight lines

1. **Nothing under `neosr/` reads argv or TOML.** argv is parsed once in each entry `main()`; `net_opt()`, `utils/rng.rng()` and `utils/options.parse_options()` are deleted.
2. **Scale is a constructor argument injected by `build_network(cfg, scale=...)`** using an explicit lazy table `name -> (module, scale_kw)`. Ported archs keep their native kwarg names; module-level `upscale, __ = net_opt()` becomes a literal default `4`.
3. **Explicit lazy tables replace `Registry` + decorators + folder scanning + star-imports.** Only the selected module is imported; the table is the single source and the list of supported components.
4. **`neosr/config.py`: one pure `load_config()`** (TOML -> normalized dict, raises `ConfigError`). The dict interface stays, so every `opt[...]` read and every shipped TOML keeps working. Framework sections are strict (unknown key -> error with did-you-mean); constructor sections are passed through and a ctor `TypeError` becomes a `ConfigError` naming the section.
5. **`neosr/runtime.py`: `setup_runtime()`** owns init_dist, seeding, backend flags and env, called once per entry point, and returns a frozen `Runtime(device, rank, world_size, seed)`.
6. **RNG:** one process-wide numpy `Generator` in `utils/rng.py`, reseeded **in place** by `seed_all()` (main) and `worker_init_fn` (from `get_worker_info().seed`). The DataLoader owns a CPU `torch.Generator`, and the sampler is seeded from `manual_seed + epoch` on the CPU, identically on all ranks.
7. **No global default device.** `torch.set_default_device("cuda")` is removed from all three entry points. Devices come from `Runtime.device` and `x.device`/`x.new_*`. Only the sites flagged by the probe and tests are fixed.
8. **Library code raises and only `main()` exits:** `ConfigError` -> 2, `KeyboardInterrupt` -> 130 (after the latest checkpoint is saved), anything else -> traceback and 1. Bare `except:` is banned.

## 1. How the three proposals' conflicts were settled

| Question | Opus | Fable | Codex | **Chosen** | Evidence / why |
|---|---|---|---|---|---|
| How the scale reaches the ctor | inject if the ctor signature accepts it | table `scale_kw`, injected in `build_network` | table `scale_arg`, injected during config normalization | **table + injection in `build_network`** | Signature inspection fails: 40 factories are `def dat_s(**kwargs)` (`dat_arch.py:1081`), so `inspect.signature` cannot see `upscale`. Injecting at build time instead of at normalization covers every construction path with one code path: train, EMA rebuild (`image.py:78`), test and `convert.py` (`convert.py:76` currently special-cases omnisr). A config that already gives the key must agree with the scale; a conflicting value is an error (Codex). |
| Registry | (not covered) | delete `Registry`; the table is the only source | keep the decorators and add manifests | **delete; the table is the only source** | Two sources of truth would need a consistency test to stay in sync. The current suffix feature (`registry.py:_do_register(suffix=)`) has zero users (grep `suffix=` in `neosr/`: none). An out-of-tree component costs one table row, the same as a decorator plus import. A test still asserts that every row resolves, so the import-all coverage Codex wants is kept. |
| RNG ownership | one module Generator, reseeded in place | `dataset.rng` attribute, set by `worker_init_fn` | `RandomStreams(python, numpy)` owned per dataset, plus `derive_seed` | **one module Generator, reseeded in place (Opus)** | Consumers use a module-global `rng`: 31 call sites in `degradations.py`, 12 in `augmentations.py`, 8 in `otf_dataset.py` and 8 in `models/otf.py`, often in nested ported helpers. Ownership would mean threading `rng=` through about 60 call sites and their callers. The process-global pattern is already how `random` and `torch` work, and torch already reseeds both per worker (`torch/utils/data/_utils/worker.py:257-265`). With in-place reseeding, every `from neosr.utils.rng import rng` binding sees the new state. Each worker process serves exactly one DataLoader, so per-dataset streams buy nothing today. Checkpointing gets one object to save (`rng.bit_generator.state`). |
| Seed when `manual_seed` is absent, under DDP | — | each rank `randint` | rank 0 picks it and broadcasts | **rank 0 picks and broadcasts** | Today `options.py:189-192` picks the seed independently per rank. The sampler seed must be identical on all ranks once the sampler uses it. |
| Unknown config keys | — | hard error | warn during migration, strict later | **hard error in framework sections; ctor sections are left to the ctor** | Today typos are silently ignored, the worst failure mode. All shipped TOMLs are a test fixture, and the did-you-mean message makes a private-config fix one line. The schema is generated from the TOML key inventory ∪ an AST scan of `opt[...]`/`.get()` reads, so a key the code reads cannot be rejected. |
| `total_iter` | — | move `logger`->`train` | alias `logger`->`train`, change the default | **keep `[logger].total_iter`; delete the dead `image.py:159`** | All 85 TOMLs that set it use `[logger]`. The only readers are `train.py:154` and `logger.py:118`, both on `[logger]` with default 1e6. `image.py:159` (`train.total_iter`, default 200000) is assigned and never read (grep `self.total_iter`: one hit). There is no conflict to migrate. |
| `[degradations]` namespace | — | merged into `datasets.train` | one normalized structure | **merged into `datasets.train` inside `load_config`; every reader uses that one** | `train.py:131` merges at runtime, yet `otf.py:146,153,207,229` read `poisson_scale_range`, `jpeg_compress` and `jpeg_range{,2}` from the **root**: a latent KeyError, or default-only behavior. The other otf.py reads already use `datasets.train`. |
| Default device | remove if the smoke test passes, else keep it scoped to the entry points | remove | remove | **remove** | Measured on the 4090 (`design/bench_optim.out`): `AdamW.step` for esrgan (702 params) takes 3.7–4.5 ms with the default device off and 17.7 ms with it on, because `adam.py:174` creates `torch.tensor(0.0)` on CUDA, followed by `.item()` per param. fwd+bwd+opt: esrgan 146–166 -> 221–238 ms/iter, swinir_small 67–75 -> 133–154. The probe over all registered archs at scale 4, train+eval, with no default device (`probe_device2.out`) fails only on `grformer_arch.py:237` and `krgn_arch.py:18`. A scoped `with torch.device` keeps both the overhead and the ambient coupling. |
| Deterministic toggle | — | keep rule "manual_seed => deterministic" | add an explicit override key | **keep the existing rule; add no key** | No config asks for it, so it would be a new key without a requirement. Easy to add later. |
| Ctor kwarg validation | — | ctor `TypeError` -> `ConfigError` | validate against the ctor signature, including the factories' fixed args | **ctor `TypeError` -> `ConfigError`; R1 removes `**kwargs` swallowing where the body never reads it** | Same safety without a signature-introspection layer. `ninasr_arch.py:119,124` and `sebica_arch.py:186` take `**kwargs` and ignore it. That drops the scale and any typo. |

## 2. Evidence the design relies on (verified)

- `arch_util.py:13-29` `net_opt()` runs `parse_options()` (argparse + TOML + `init_dist` + seeding + cudnn flags) on every call. 46 of 55 `*_arch.py` files call it at import (`upscale, __ = net_opt()` ×42 and `upscale, training = net_opt()` ×4, where `training` is unused). It is also called in `DropPath.__init__` (`arch_util.py:209-210`, where the dead flag is overwritten by `.train()/.eval()`) and in `losses/vgg_perceptual_loss.py:16`, which picks PatchLoss kernels. Importing any arch without `-opt` raises `ValueError`.
- Native scale kwargs (AST, `design/tables.py`, 99 arch rows):
  - `scale`: asid, cugan, desrgan, eimn, eqrsrgan, esrgan, flexnet, krgn, man, moesr, mosrv2, ninasr, pancam, plainusr, rcan.
  - `upscale`: atd, catanet, cfsr, compact, craft, dat, dct, dctlsa, ditn, drct, grformer, hasn, hat, hitsrf, hma, mfghmoe, microsr, rgt, span, spanplus, srformer, swinir, swinirdense.
  - `upscaling_factor`: esc, lmlt, plksr, realplksr, safmn.
  - `upscale_factor`: msdan. `sr_rate`: sebica. `upsampling`: omnisr (read via `kwargs["upsampling"]`, `omnisr_arch.py:1057`; the generator emits `None`, so the row is fixed by hand).
  - Five factories drop the scale: `eimn_arch.py:313,318`, `ninasr_arch.py:119,124` and `sebica_arch.py:186`. Registry names are not file stems (`pancam_arch.py` registers `pancamsr`).
- `utils/rng.py:9-14` parses argv + TOML. Ten module-level `rng = rng()` sites; the four live ones are `data/augmentations.py:10`, `data/degradations.py:14`, `data/otf_dataset.py:24` and `models/otf.py:20`. Spawned workers (`train.py:682`) re-import, re-parse argv and get **identical** Generators. `data/__init__.py:53-59` `worker_init_fn` reseeds `random`/`torch`/legacy numpy with `num_workers*rank+worker_id+seed`. That seed does not depend on the epoch, so it overrides torch's per-iterator `base_seed+worker_id` seeding and repeats sequences every epoch. It never touches the Generator that the code actually uses.
- `data_sampler.py:40-42` seeds the sampler with `epoch` only and builds the generator **on CUDA**.
- `degradations.py:703,706` call `rng.randn` on a `numpy.random.Generator`, which has no `randn`: a latent `AttributeError` (Codex).
- `options.py:181-216` mixes `init_dist`, seed choice, `set_random_seed(seed+rank)`, cudnn/deterministic flags and env vars. Setting `PYTHONHASHSEED` inside the running interpreter has no effect (Codex).
- Default device: `train.py:320`, `test.py:19` and `convert.py:298`. Non-arch code depending on it: `image.py:542` `torch.zeros(1)`; `consistency_loss.py:42` (plain tensor attribute, not moved by `.to()`) and `:132`; `wavelet_guided.py:18-36`; `msswd_loss.py:14`; `metrics/calculate.py:198-202,235-239` (metric nets built per image and never moved); `convert.py:138-140,175-177`; `train.py:406,410`; and `data_sampler.py:40` (explicit CUDA). `base.py:27`, `otf.py:30,57,65` and `prefetch_dataloader.py:86-88` hard-code `cuda`.
- Registries: `archs/__init__.py:15-27` re-scans and imports all archs on **every** `build_network`. data/losses/models/metrics import every module at package import, and `neosr/__init__.py` star-imports everything. The `common/` path hack (`train.py:673-679`) has no importers.
- `sys.exit`: 37 AST sites in `neosr/` plus the entry scripts. Among them, `base.py:407` and `:445` exit with **0** on a checkpoint load failure, and `train.py:579-584` kills a run for low throughput. There are also 13 bare `except` clauses.
- Configs: 135 TOMLs (93 image, 42 otf). `options/train_dctlsa.toml:88` is invalid TOML (duplicate `type` in `[train.ldl_opt]`).

## 3. Code

### 3.1 `neosr/archs/__init__.py` (same shape for `losses`, `data`, `models`, `metrics`)

```python
import importlib
from collections.abc import Mapping
from typing import Any
from torch import nn
from neosr.config import ConfigError

# name -> (module under neosr.archs, ctor kwarg that receives the run scale or None)
# Generated once by design/gen_tables.py, then maintained by hand. One row per class AND per factory.
ARCHS: dict[str, tuple[str, str | None]] = {
    "compact": ("compact_arch", "upscale"), "dat_s": ("dat_arch", "upscale"),
    "esrgan": ("esrgan_arch", "scale"), "plksr": ("plksr_arch", "upscaling_factor"),
    "msdan": ("msdan_arch", "upscale_factor"), "sebica": ("sebica_arch", "sr_rate"),
    "omnisr": ("omnisr_arch", "upsampling"), "pancamsr": ("pancam_arch", "scale"),
    "unet": ("unet_arch", None), "metagan": ("metagan_arch", None),  # ... 99 rows
}

def build(table: Mapping[str, tuple[str, str | None]], package: str, section: str,
          cfg: Mapping[str, Any], *, scale: int | None = None, **extra: Any) -> Any:
    """Shared by build_network/build_loss/build_metric: lazy import, scale injection, error mapping."""
    kwargs = dict(cfg)
    name = kwargs.pop("type", None)
    if name not in table:
        raise ConfigError(f"[{section}] unknown type {name!r}", key=name, known=table)
    module, scale_kw = table[name]
    obj = getattr(importlib.import_module(f"{package}.{module}"), name)   # ImportError propagates, with context
    if scale_kw is not None:
        if scale is None:
            raise ConfigError(f"[{section}] {name} needs the run scale")
        if kwargs.setdefault(scale_kw, scale) != scale:
            raise ConfigError(f"[{section}] {scale_kw}={kwargs[scale_kw]} contradicts scale={scale}")
    try:
        return obj(**kwargs, **extra)
    except TypeError as e:          # bad/unknown ctor kwarg from TOML
        raise ConfigError(f"[{section}] {name}: {e}") from e

def build_network(cfg: Mapping[str, Any], *, scale: int | None = None) -> nn.Module:
    return build(ARCHS, "neosr.archs", "network", cfg, scale=scale)
```

`build` lives in `neosr/utils/factory.py`, so the five packages share it. Callers use `build_network(opt["network_g"], scale=opt["scale"])`, the same for EMA (`image.py:78`), `build_network(opt["network_d"])` (`scale_kw` None, so the scale is ignored), and in `convert.py` `build_network({"type": args.network, **win}, scale=args.scale)`. That last call deletes the omnisr special case. `LOSSES` has `"vgg_perceptual_loss": ("vgg_perceptual_loss", "scale")`, and every other row is `None`. `MODELS` = `{"image": ("image", None), "otf": ("otf", None)}`; legacy models are removed by the structure topic. The model is built by `build_model(opt, runtime)`.

The `try/except TypeError` wraps only the constructor call. A `TypeError` raised *inside* a ctor body is also mapped to `ConfigError`, but `from e` keeps the traceback, so nothing is hidden.

### 3.2 Arch file after recipe R1 (example: `dat_arch.py`)

```python
# before                                            # after
from neosr.archs.arch_util import net_opt, ...      from neosr.archs.arch_util import ...
from neosr.utils.registry import ARCH_REGISTRY      # (deleted)
upscale, __ = net_opt()                             # (deleted)
@ARCH_REGISTRY.register()                           # (deleted)
class dat(nn.Module):                               class dat(nn.Module):
    def __init__(self, ..., upscale: int = upscale,     def __init__(self, ..., upscale: int = 4,
@ARCH_REGISTRY.register()                           # (deleted)
def dat_s(**kwargs): return dat(depth=..., **kwargs) def dat_s(**kwargs): return dat(depth=..., **kwargs)
```

### 3.3 `arch_util.py` and `vgg_perceptual_loss.py`

```python
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        super().__init__()
        self.drop_prob, self.scale_by_keep = drop_prob, scale_by_keep
    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

# DySample.forward
coords_h = torch.arange(H, device=x.device, dtype=x.dtype) + 0.5
coords_w = torch.arange(W, device=x.device, dtype=x.dtype) + 0.5
coords = torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij")).transpose(1, 2)[None, :, None]
normalizer = x.new_tensor([W, H]).view(1, 2, 1, 1, 1)        # pin_memory on a CUDA tensor removed
# net_opt() deleted.

# vgg_perceptual_loss.py
class vgg_perceptual_loss(nn.Module):
    def __init__(self, ..., patchloss: bool = False, scale: int = 4, ...):  # scale injected via LOSSES row
        if patchloss and scale not in PATCH_KERNELS:   # validate before building VGG
            raise ValueError(f"patchloss does not support scale={scale}")
```

### 3.4 `neosr/config.py`

```python
import difflib, re, tomllib
from pathlib import Path
from typing import Any

class ConfigError(ValueError):
    def __init__(self, msg: str, *, key: str | None = None, known=()):
        if key and (m := difflib.get_close_matches(key, list(known), n=1)):
            msg += f" (did you mean {m[0]!r}?)"
        super().__init__(msg)

REQUIRED = object()
ROOT_KEYS = {"name": (str, REQUIRED), "model_type": (str, REQUIRED), "scale": (int, REQUIRED),
             "num_gpu": (int | str, "auto"), "manual_seed": (int, None), "use_amp": (bool, False),
             "bfloat16": (bool, False), "fast_matmul": (bool, False), "compile": (bool, False),
             "find_unused_parameters": (bool, False), "network_g": (dict, REQUIRED), "network_d": (dict, None),
             "datasets": (dict, REQUIRED), "train": (dict, None), "val": (dict, None), "logger": (dict, {}),
             "path": (dict, {}), "degradations": (dict, None), "dist_params": (dict, None)}
# Framework sections: set of allowed keys, generated by toml_keys.py ∪ AST scan of opt reads; reviewed once.
SECTIONS: dict[str, frozenset[str]] = {"datasets.*": ..., "train": ..., "logger": ..., "val": ...,
                                       "path": ..., "degradations": ...}
# Constructor-owned subtrees: not schema-checked here; the ctor rejects unknown kwargs (build -> ConfigError).
OPAQUE = re.compile(r"^(network_[gd]|dist_params|train\.(\w+_opt|optim_[gd]|scheduler)|val\.metrics\.\w+)$")

def load_config(path: str | Path, *, is_train: bool, root: Path, debug: bool = False,
                auto_resume: bool = False) -> dict[str, Any]:
    """Pure: no argv, no CUDA, no RNG, no mkdir, no process-group."""
    try:
        cfg = tomllib.loads(Path(path).read_text("utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from e
    _check(cfg, "", ROOT_KEYS)          # types, REQUIRED, defaults, unknown keys (walks SECTIONS, skips OPAQUE)
    _derive(cfg, root=root, is_train=is_train, debug=debug, auto_resume=auto_resume)
    return cfg

def _derive(cfg, *, root, is_train, debug, auto_resume):
    # options.py:214-276 moved verbatim (minus side effects), plus:
    #   - per dataset: phase, scale; "datasets.train" gets cfg["degradations"] merged (was train.py:131)
    #   - expanduser on paths; experiments_root/results_root/models/log/visualization paths derived ONCE
    #     (train.py:329 no longer overwrites them)
    #   - debug: name "debug_" prefix + freq overrides; is_train, auto_resume copied in
    ...

def validate_train(cfg: dict[str, Any]) -> None:
    """Cross-key checks, before any GPU allocation: the 14 image.py:310-433 sys.exit checks,
    train.py:183/488/511, patch_size % 4, persistent_workers requires num_worker_per_gpu > 0,
    GAN loss requires network_d and optim_d, etc. Raises ConfigError naming the dotted key."""
```

Strictness: a section in `SECTIONS` rejects unknown keys. A wrong type is rejected as well: bool where int is wanted, since `isinstance(True, int)` must be special-cased. `OPAQUE` subtrees are owned by their constructors. The "derived" keys (`dist`, `rank`, `world_size`, `num_gpu` resolved, `is_train`, `auto_resume`, path roots) stay in the dict, so no reader changes. Typed config structs are not introduced.

### 3.5 `neosr/runtime.py`

```python
import os, random
from dataclasses import dataclass
import numpy as np, torch
from neosr.utils.dist_util import get_dist_info, init_dist
from neosr.utils.rng import seed_rng

@dataclass(frozen=True, slots=True)
class Runtime:
    device: torch.device
    rank: int
    world_size: int
    seed: int
    @property
    def distributed(self) -> bool: return self.world_size > 1

def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed); seed_rng(seed)

def setup_runtime(cfg: dict, *, launcher: str = "none") -> Runtime:
    deterministic = cfg.get("manual_seed") is not None
    if deterministic:                                      # must precede the first cuBLAS handle
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    if launcher != "none":
        init_dist(launcher, **(cfg.get("dist_params") or {}))
    rank, world = get_dist_info()
    seed = cfg.get("manual_seed")
    if seed is None:                                       # one seed for all ranks (sampler needs it)
        box = [random.randint(1024, 10000) if rank == 0 else None]
        if world > 1: torch.distributed.broadcast_object_list(box, src=0)
        seed = box[0]
    seed_all(seed + rank)
    torch.backends.cudnn.deterministic = deterministic     # set BOTH branches explicitly
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.utils.deterministic.fill_uninitialized_memory = deterministic
    # + fast_matmul/tf32 flags from train.py:342-346, verbatim
    if cfg.get("bfloat16") and torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("bfloat16 requires compute capability >= 8.0")
    device = torch.device("cuda", torch.cuda.current_device())
    cfg.update(dist=launcher != "none", rank=rank, world_size=world, manual_seed=seed,
               deterministic=deterministic,
               num_gpu=torch.cuda.device_count() if cfg.get("num_gpu", "auto") == "auto" else cfg["num_gpu"])
    return Runtime(device, rank, world, seed)
```

### 3.6 RNG: `neosr/utils/rng.py` and `neosr/data/__init__.py`

```python
# neosr/utils/rng.py — no imports from neosr
import numpy as np

rng: np.random.Generator = np.random.default_rng()
"""Process-wide numpy stream for data/degradation code. Reseeded in place (never rebound), so every
`from neosr.utils.rng import rng` sees the new state. Seeded by runtime.seed_all (main) and
data.worker_init_fn (workers)."""

def seed_rng(seed: int) -> None:
    rng.bit_generator.state = np.random.PCG64(seed).state
```

```python
# neosr/data/__init__.py
def worker_init_fn(_: int) -> None:
    # torch already seeded random / torch / legacy numpy from base_seed + worker_id
    seed_rng(torch.utils.data.get_worker_info().seed)

def build_dataloader(ds, cfg, *, runtime: Runtime, sampler=None) -> DataLoader:
    train = cfg["phase"] == "train"
    workers = _num_workers(cfg, runtime) if train else 0
    g = torch.Generator().manual_seed(runtime.seed + runtime.rank)   # loader-owned; independent of model-side draws
    return DataLoader(
        ds, batch_size=_batch_size(cfg, runtime) if train else 1, shuffle=train and sampler is None,
        sampler=sampler, num_workers=workers, drop_last=train, generator=g, worker_init_fn=worker_init_fn,
        pin_memory=cfg.get("pin_memory", True),
        persistent_workers=cfg.get("persistent_workers", True) and workers > 0,
        prefetch_factor=2 if workers else None,
    )   # one args dict for DDP and non-DDP; fixes the UnboundLocalError path at :99
```

```python
# data_sampler.py EnlargedSampler.__iter__
g = torch.Generator()                                   # CPU; same on every rank
g.manual_seed(self.seed + self.epoch)                   # seed = cfg["manual_seed"] (rank-independent)
indices = torch.randperm(self.total_size, generator=g).tolist()
```

Consumer change (recipe R2): `from neosr.utils.rng import rng` stays, `rng = rng()` is deleted, and `rng.randn(*shape)` becomes `rng.standard_normal(shape)` at `degradations.py:703,706`. Crops keep using `random.randint`, which has inclusive bounds; do not convert them to `rng.integers`. Semantics:
- `num_workers == 0`: the main-process stream from `seed_all`.
- Persistent workers: each worker is seeded once and its stream continues across epochs, distinct per worker.
- Non-persistent workers: reseeded each epoch from the advancing loader generator.
- Same seed, worker count and persistence setting give the same stream.

Exact mid-epoch resume, which needs RNG, loader generator, sampler cursor and OTF queue state, belongs to the checkpoint topic. T3 provides the state objects: `rng.bit_generator.state`, `torch.get_rng_state()`, `g.get_state()`, `sampler.epoch`.

### 3.7 Devices

- Remove `torch.set_default_device("cuda")` from `train.py`, `test.py` and `convert.py`. Never replace it with `set_default_device("cpu")` or `with torch.device(...)`.
- Models are constructed on the CPU and moved once by `model_to_device`. `base.py:27`, `otf.py:30,57,65`, `prefetch_dataloader.py` (streams, `current_stream`) and `torch.load(map_location=...)` use `runtime.device`.
- Rules inside modules:
  - Values derived from data: `x.new_tensor(...)`, `x.new_zeros(...)` and `torch.*_like(x)`.
  - Index tensors: `arange(..., device=x.device)`.
  - Constants built in a ctor: `register_buffer(..., persistent=False)`, so `.to()` moves them without changing checkpoint keys.
  - Modules created at deploy time: `device=w.device, dtype=w.dtype`.
- Known sites to fix:
  - `grformer_arch.py:237`: `x.new_tensor(1 / 0.01).log()`.
  - `krgn_arch.py:18-22`: drop `.to("cuda")` and build the eye on `x.device`.
  - `image.py:542`: `l_g_total = self.output.new_zeros(())`.
  - `consistency_loss.py:42`: buffer. `consistency_loss.py:132`: `img.new_tensor(...)`.
  - `wavelet_guided.py:18-36`: `device=x.device`.
  - `msswd_loss.py:14`: `device`/`dtype` from input.
  - `metrics/calculate.py`: build DISTS/TOPIQ once per validation, on `runtime.device`, with lazy import.
  - `convert.py:138-140,175-177` and `train.py:406,410`: `device=rt.device`.
  - `plainusr_arch.py:73` deploy path: `device`/`dtype` from weights.
- Do **not** add `device=` to the roughly 335 device-less factory calls that work: CPU constants in `__init__` that get moved, or values passed as Python scalars. The arch and loss test matrix decides which sites need it.

### 3.8 Exceptions and entry points

```python
# train.py (test.py and convert.py have the same shape)
def main(argv: list[str] | None = None) -> int:
    args = cli.train_args(argv)                       # argparse may SystemExit(2) on bad CLI: allowed
    try:
        cfg = load_config(args.opt, is_train=True, root=ROOT, debug=args.debug, auto_resume=args.auto_resume)
        validate_train(cfg)
        rt = setup_runtime(cfg, launcher=args.launcher)
        train(cfg, rt)                                 # KeyboardInterrupt: save latest in train(), re-raise
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr); return 2
    except KeyboardInterrupt:
        return 130
    except Exception:
        logging.getLogger("neosr").exception("training failed"); return 1
    finally:
        if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()   # no barrier
    return 0

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    sys.exit(main())
```

Policy:
- `sys.exit`/`SystemExit` is allowed only in `if __name__ == "__main__"` and argparse.
- Library exception types: `ConfigError` for config, `ValueError`/`TypeError` for direct API misuse, native `OSError` subclasses for I/O (with the path), and `RuntimeError` with context for failed operations.
- Log once, where the error is handled. Keep the cause with `raise ... from e`.
- Bare `except:` is banned; use ruff `E722` and `BLE001` scoped to `neosr/`. In library code, `print(` becomes the logger.

sys.exit map (recipe R3):

| Site | Becomes |
|---|---|
| `image.py:310-433` (14) | `validate_train` -> `ConfigError` (before GPU allocation) |
| `base.py:199,227` | `ConfigError` |
| `base.py:377,499` | `OSError` after the retries |
| `base.py:407` | `FileNotFoundError` (was exit **0**) |
| `base.py:413-445` | delete the bare except; `load_state_dict` raises (was exit 0) |
| `otf_dataset.py:58` | `ConfigError` |
| `otf_dataset.py:166`, `paired_dataset.py:129` | `OSError(f"cannot read {path}")`. The DataLoader re-raises worker exceptions; `sys.exit` only killed the worker. |
| `transforms.py:122,127` | `ValueError` |
| `lmdb_util.py:106` | `ValueError` |
| `lmdb_util.py:199` | `FileExistsError` |
| `train.py:183,488,511` | `validate_train` |
| `train.py:474` | dead, delete |
| `train.py:579-584` (throughput kill) | delete |
| `train.py:597` | `RuntimeError` after save |
| `train.py:636,642`, `test.py:68` | `main()` |
| `options.py:38` | deleted with the module |

### 3.9 Imports

- `neosr/__init__.py` is empty: no star-imports. `neosr/utils/__init__.py` stops re-exporting `toml_load` and options, which breaks the options <-> utils cycle.
- Package `__init__`s contain only the table and `build_*`. They import no implementation modules.
- Heavy optional dependencies are imported inside the implementation that needs them: `calculate.py:9` DISTS/TOPIQ, `torchprofile`.
- Delete the `common/` `sys.path` hack (`train.py:673-679`).
- `tests/test_imports.py` imports every module in a fresh interpreter. That keeps "lazy at runtime" from hiding a broken module in CI.

## 4. Mechanical recipes (one file per agent invocation)

**R1: arch file with `net_opt`** (46 files). Use AST, never global text replacement of `scale`/`training`.
1. Delete `net_opt` from the `arch_util` import and the `from neosr.utils.registry import ARCH_REGISTRY` line.
2. Delete the module-level `X, Y = net_opt()` line. Find every remaining reference to `X` and `Y` (AST `Name` loads).
3. Replace each ctor default `= X` with `= 4`, keeping the parameter name and annotation (`int`). If `X`/`Y` is referenced anywhere else (a function body, class attribute or helper default), stop and report the file.
4. Delete every `@ARCH_REGISTRY.register()` line.
5. Every factory must end in `**kwargs` forwarded to the class it instantiates, and must not hard-code the scale kwarg. Fix `eimn_l/eimn_a` (`eimn_arch.py:313,318`), `ninasr_b0/b2` (`:119,124`) and `sebica_mini` (`:186`).
   - 5b. If a class `__init__` takes `**kwargs` and its body never reads `kwargs`, delete it. omnisr reads `kwargs["upsampling"]`, so it keeps `**kwargs`.
6. Check that each class and factory name in the file has a row in `ARCHS`, with the right `scale_kw`.
7. Verify (step 7 of the gate in §6):
   - `.venv/bin/python -I -c "import sys; sys.argv=['x']; import neosr.archs.<stem>"`
   - `pytest tests/test_archs.py -k "<names>"`
   - On a device-mismatch failure, fix only the flagged call with `x.device`/`x.new_*`.

**R1b: arch file without `net_opt`** (9 files): steps 1 (registry only), 4 and 6.

**R2: RNG consumer** (`augmentations.py`, `degradations.py`, `otf_dataset.py`, `models/otf.py`): delete `rng = rng()`; change `rng.randn(*s)` to `rng.standard_normal(s)`; no other change.

**R3: `sys.exit` site**: use the table in §3.8. The replacement exception carries the same message plus the offending path or key.

**R4: package `__init__`**: replace scan + decorator import with the table and `build_*` from §3.1. Delete `@*_REGISTRY.register()` and registry imports in member modules. Check the rows with the `test_registry_rows` test.

## 5. Migration order

- **S0, serial: new files only.**
  - `neosr/config.py`, `neosr/runtime.py`, `neosr/cli.py` (the three argparse parsers), `neosr/utils/factory.py`, and the new `utils/rng.py` content. The old `rng()` stays importable as a shim until S1e: `def rng(): return _rng`.
  - Tables generated by `design/gen_tables.py`, with omnisr fixed by hand.
  - Tests: `test_imports`, `test_config`, `test_archs`, `test_losses`, `test_data`, `test_hygiene` (see §6).
  - Baseline: `test_config` passes except `train_dctlsa.toml`, which is fixed by deleting the duplicate key at line 88. `test_imports` fails for the `net_opt` files, and that is expected until S1a.
- **S1, parallel: disjoint file sets.**
  - **S1a** — 46 arch files, R1, split into batches.
  - **S1b** — 9 non-scale arch files, R1b.
  - **S1c** — `arch_util.py` (DropPath, DySample, delete `net_opt`) plus the five package `__init__`s (R4).
  - **S1d** — `losses/*`, `metrics/*`: registry removal, the `vgg_perceptual_loss` scale kwarg, and the device fixes in §3.7.
  - **S1e** — `data/*`: R2, `worker_init_fn`, `build_dataloader`, the sampler, prefetcher device and no preload in `__init__` (`train.py:541-543` currently discards a batch), R3 sites, then delete the `rng()` shim.
  - **S1f** — `models/base.py`, `image.py`, `otf.py`: `runtime` parameter and device, R3, `l_g_total`, delete `image.py:159`, root degradation reads to `datasets.train`, `build_network(..., scale=)` (including EMA).
  - **S1g** — `train.py`, `test.py`, `convert.py`: `main()`, removal of the default device, the `common/` hack and the throughput kill, and `convert.py` scale from `-s`.
- **S2, serial: integration.**
  - Run the full `pytest`.
  - 30-iteration `--debug` runs: one paired config (`train_span.toml`), one otf config, and one GAN config.
  - Save, resume, then 10 more iterations, followed by `test.py` and `convert.py` on the saved model.
  - The performance check in §6.
- **S3, serial: delete obsolete code.**
  - `utils/options.py`, `utils/registry.py`, star-imports, `optimizers/soap_sf.py:8-20` dead text.
  - `test_hygiene` must be green.

S1 sub-steps touch disjoint files. S1f and S1g depend only on S0 interfaces. Only S2 needs everything.

## 6. Verification (all runnable on one 4090)

| Test | Asserts |
|---|---|
| `tests/test_imports.py` | Every `neosr.*` module imports in a fresh subprocess with `sys.argv=["x","--bogus"]` and no `-opt`. `torch.cuda.is_initialized()` is False afterwards, no file is created, and `torch.get_default_device()` is cpu. |
| `tests/test_hygiene.py` | grep `neosr/` for `net_opt`, `parse_options`, `set_default_device`, `sys.exit`, `sys.argv`, `except:` and `_REGISTRY`: zero hits. |
| `tests/test_registry.py` | Every row in all five tables resolves to a module attribute. Every `scale_kw` is a parameter of the target class's `__init__`; for factories, of the class it calls. |
| `tests/test_config.py` | All shipped TOMLs `load_config` + `validate_train` without error. An unknown key gets `ConfigError` with a suggestion. A bool given for an int is rejected. A conflicting `upsampling` gets `ConfigError`. `[degradations]` keys appear in `datasets.train`. |
| `tests/test_archs.py` | For each `ARCHS` row with `scale_kw`: build at scale 2 then 4 in one process. CUDA forward on a 2×3×64×64 input, train and eval, no default device; `out.shape[-1] == 64*scale`, plus backward in train. DropPath: train is stochastic, eval is the identity, and `.train()` after `.eval()` restores dropping. Fixed-weight check: `state_dict` keys and shapes match a pre-migration dump, generated once at S0 from the old code. |
| `tests/test_losses.py` | Every loss is built on CUDA with no default device and gives a finite scalar plus a gradient to the prediction. `vgg_perceptual_loss(patchloss=True)` works at scales 2 and 4. |
| `tests/test_data.py` | Two workers draw different `rng` values. Identical settings run twice give identical first batches. Epoch 2 is not equal to epoch 1, with `persistent_workers` both on and off. `num_workers=0` works. Sampler: same permutation for rank 0 and rank 1 before sharding, different across epochs, and no CUDA allocation. |
| `tests/test_errors.py` | A missing pretrain path raises `FileNotFoundError`. `main()` with malformed TOML returns 2. An unreadable image in a worker surfaces in the parent with its path. |
| Perf gate | Steady-state ms/iter before and after on `span`, `esrgan` and `swinir_small` (`design/bench_fwd.py`). The expected direction is set by the measured numbers in §1: after must not be slower. |

## 7. Risks

- **Changed RNG streams.** Worker streams were identical across workers and epochs, sampler order ignored the seed, and models were initialized under the default device. Loss curves for the same seed will differ from pre-migration. The gate is "runs, saves, resumes, converts", not bitwise equality, and fixed-weight outputs must match.
- **Strict schema vs. private configs.** One-line fix with a suggestion. If the owner prefers, `ConfigError` for unknown keys can become a warning for one release by flipping one function.
- **Coverage gaps in the probe.** Scale-2-only branches, tiling and deploy paths are not covered by it, so the matrix tests are the gate. hit_srf/esc need 64 px inputs, dat needs batch ≥ 2 in train, esc_fp needs a config that is not tested.
- **DDP.** No second GPU is available here, so DDP and seed-broadcast paths get design and code review only. DataParallel requires the primary device to be `cuda:0`; `current_device()` satisfies that.
- **`try/except TypeError` around ctor calls.** It also catches TypeErrors raised inside a ctor body. That is acceptable because the message names the section and `from e` keeps the traceback.

## 8. Deliberately not done

- Typed config structs (msgspec/pydantic as in traiNNer `redux_options.py`).
- A `Config` singleton (traiNNer `config.py:19`), a cached `net_opt()`, or a contextvar carrying the run.
- Renaming 46 ported ctor scale params, or introspecting signatures.
- Per-owner `RandomStreams` and `rng=` plumbing. Revisit only if one process ever hosts two stochastic datasets.
- A scoped default device. Adding `device=` to calls that work.
- try/except auto-discovery or silent import fallbacks.
- An explicit `deterministic` key.
- Exact mid-epoch resume (checkpoint topic).
- The loss-call contract (T1, which owns how each loss pulls its inputs and the discriminator). T3 only supplies `build_loss(cfg, scale=)` and device rules.

## 9. Open questions for the owner

1. Unknown keys in framework sections: hard error (chosen) or a warning for one release?
2. Should `convert.py` keep `-s/--scale` as its only scale source (chosen), or also accept `-opt`?
3. OK to fix `options/train_dctlsa.toml:88` (duplicate `type`) as part of S0?
