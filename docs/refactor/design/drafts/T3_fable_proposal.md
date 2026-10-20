# T3 draft: configuration, construction, imports, RNG, devices, error handling (neosr-dev)

Repo: /home/korvin/apps/neosr-dev (python 3.13, torch 2.9.1). Paths below are relative to it.

## Current behavior relied on (evidence)
- neosr/archs/arch_util.py:13-29 `net_opt()` builds an argparse parser and parses sys.argv + TOML every call (via neosr/utils/options.py:41-280 `parse_options`). 46 of 55 `*_arch.py` call it at module import (`upscale, __ = net_opt()` x42, `upscale, training = net_opt()` x4; the `training` value is never read anywhere else). The class ctor default is `upscale: int = upscale` (or `scale`, `upscaling_factor`, `upscale_factor` (msdan), `sr_rate` (sebica), `upsampling` (omnisr)). Factory variants (`dat_s`, `hat_l`, `swinir_small`, ... 40 of them) are `def x(**kwargs): return cls(fixed..., **kwargs)` and rely on the class default for scale.
- arch_util.py:205-214 `DropPath.__init__` calls `net_opt()` per instance and assigns `self.training` — which `nn.Module.train()/eval()` overwrite anyway (image.py:119 `net_g.train()`, image.py:891 `model.eval()`); the config flag is dead.
- neosr/losses/vgg_perceptual_loss.py:16 `upscale, __ = net_opt()`, used at :118-125 to pick PatchLoss kernel sizes.
- neosr/utils/rng.py:9-14 `rng()` parses argv+TOML and returns a `numpy.random.Generator`; module-level `rng = rng()` in data/augmentations.py:10, data/degradations.py:14, data/otf_dataset.py:24, models/otf.py:20 (+ legacy eqrotf/pancam/simple_otf datasets and models). DataLoader workers are spawned (train.py:682 `set_start_method('spawn')`), re-import these modules, re-parse argv, and build the *same-seeded* Generator in every worker; data/__init__.py:53-59 `worker_init_fn` reseeds only `random`/`torch`/`np.random` legacy, which torch already does itself (torch/utils/data/_utils/worker.py:258-265 seeds random, torch and numpy legacy from `base_seed + worker_id`).
- options.py:182-213: `parse_options` side effects: `init_dist` (:187-189), seed choice + `set_random_seed(seed + rank)` (:193-213), cudnn/deterministic flags and env vars (:198-212). It is called from train.py:324, test.py:17, convert.py:301, arch_util.py:16, rng.py:11 (and a dead string literal in optimizers/soap_sf.py:9-20). Importing any arch without `-opt` raises `ValueError("Didn't get a config!")` (:159-165).
- train.py:320, test.py:19, convert.py:298 `torch.set_default_device("cuda")`. torch/utils/_device.py:62-108: this installs a `DeviceContext(TorchFunctionMode)` at the bottom of the torch-function stack; every `torch.*` call then goes through Python `__torch_function__` while it is active. `base.py:27` and `otf.py:33`, prefetch_dataloader.py:88 hard-code `torch.device("cuda")`; data_sampler.py:40-42 creates a CUDA generator + `randperm` on GPU then `.tolist()`.
- Registries: neosr/utils/registry.py (fvcore copy). neosr/archs/__init__.py:15-27 scans the folder and imports every `*_arch.py` on **every** `build_network` call; losses/models/data `__init__` import everything at package import; neosr/__init__.py star-imports all packages. 54 arch, 13 loss, 8 dataset, 6 model, 2 metric decorators.
- `sys.exit` inside library code: 37 sites (image.py x14 at :310-433, base.py x6 at :199,227,377,407(exit 0 on missing file),445(exit 0 on load failure),499, train.py x8, data x5, utils x3, test.py x1). 13 bare `except:`/`except Exception: pass`. 37 `print(` in neosr/.
- Config: 129 option files (86 image, 42 otf). options/train_dctlsa.toml is invalid TOML (duplicate `type` at line 88) and nobody noticed. `total_iter` is read from `[logger]` by train.py:154 and logger.py:118 but from `[train]` (default 200000) by image.py:159. `[degradations]` is merged into `datasets.train` by train.py:129-132, yet otf.py:146,154 read `poisson_scale_range`/`jpeg_range` from the root opt.

## Decisions (one-line rationale each)
D1 Scale is a constructor argument injected by `build_network(cfg, scale=...)`; nothing reads argv/TOML at import. Every arch is importable and constructible standalone.
D2 Keep each arch's native kwarg name; the registry table declares it (`scale_kw`). One token per row beats editing 46 vendored ctor bodies.
D3 Registries become explicit lazy tables in each package `__init__` (`name -> (module, scale_kw)`); the `Registry` class, decorators, folder scanning and `neosr/__init__.py` star-imports are deleted. Only the requested module is imported; a broken arch cannot break startup.
D4 `DropPath` uses `nn.Module.training`; the config flag was dead.
D5 Config: `neosr/config.py` = pure `load_config()` (TOML -> normalized dict, derived paths, key migration, validation) + `ConfigError`. Dict interface is kept (no typed structs) so all `opt[...]` reads and all TOML files keep working.
D6 Validation is strict for framework-owned sections (unknown key -> ConfigError with a did-you-mean), pass-through for constructor sections (network_*, train.*_opt, train.optim_*, train.scheduler, val.metrics.*) where the ctor's `TypeError` is re-raised as ConfigError naming the section.
D7 Runtime side effects live in `neosr/runtime.py: setup_runtime(cfg, launcher) -> Runtime(device, rank, world_size, seed, deterministic)`; called once from each entry script's `main()`; also writes the legacy derived keys (`dist`, `rank`, `world_size`, `num_gpu`, `is_train`, `manual_seed`, `deterministic`, `root_path`) into cfg for compatibility.
D8 CLI parsing lives in `neosr/cli.py` (`train_args`, `test_args`, `convert_args`); argv is parsed exactly once in `main()`.
D9 RNG: one seed; torch/random/numpy-legacy are seeded by `seed_all(seed + rank)` in the main process and by torch itself in workers; every component that draws numpy randomness owns a `numpy.random.Generator` (`self.rng`) passed explicitly to pure functions (`degradations.py` functions take `rng=`); `worker_init_fn` replaces `dataset.rng` with `default_rng(get_worker_info().seed)` so workers/epochs/ranks get distinct, reproducible streams. `neosr/utils/rng.py` is deleted.
D10 Remove `torch.set_default_device`. Device comes from `Runtime.device` and flows explicitly (`model(cfg, device)`, prefetcher, loaders); inside modules use `x.device`/`x.new_*`. Only archs/losses that the CUDA probe flags get `device=` fixes.
D11 Exceptions: library code raises (`ConfigError` for configuration, builtin types otherwise); `sys.exit` is allowed only in the `main()` of train.py/test.py/convert.py, which map ConfigError -> exit 2, KeyboardInterrupt -> save + exit 130, anything else -> traceback + exit 1. Throughput kill-switch (train.py:579-584) deleted. Bare `except:` banned.
D12 Legacy models/datasets (default.py, simple_otf, eqr_otf, pancam_otf and their datasets) are not migrated; they are deleted by the structure topic. Only `image` and `otf` are migrated.

## Code skeletons

### neosr/archs/__init__.py
```python
from __future__ import annotations
import importlib
from collections.abc import Mapping
from typing import Any
from torch import nn
from neosr.config import ConfigError

# type name -> (module under neosr.archs, kwarg that receives the scale; None = no scale)
ARCHS: dict[str, tuple[str, str | None]] = {
    "span": ("span_arch", "upscale"), "span_fast": ("span_arch", "upscale"),
    "dat_s": ("dat_arch", "upscale"), "dat_m": ("dat_arch", "upscale"), "dat_2": ("dat_arch", "upscale"),
    "esrgan": ("esrgan_arch", "scale"), "plksr": ("plksr_arch", "upscaling_factor"),
    "msdan": ("msdan_arch", "upscale_factor"), "sebica": ("sebica_arch", "sr_rate"),
    "omnisr": ("omnisr_arch", "upsampling"),
    "metagan": ("metagan_arch", None), "unet": ("unet_arch", None), "patchgan": ("patchgan_arch", None),
    # ... one row per registered name (generated once from the AST scan, then maintained by hand)
}

def build_network(cfg: Mapping[str, Any], *, scale: int | None = None) -> nn.Module:
    cfg = dict(cfg)
    name = cfg.pop("type", None)
    if name not in ARCHS:
        raise ConfigError(f"[network] unknown type {name!r}", known=ARCHS)
    module, scale_kw = ARCHS[name]
    cls = getattr(importlib.import_module(f"neosr.archs.{module}"), name)
    if scale_kw is not None:
        if scale is None:
            raise ConfigError(f"[network] {name} requires a scale")
        if cfg.setdefault(scale_kw, scale) != scale:
            raise ConfigError(f"[network] {scale_kw}={cfg[scale_kw]} contradicts scale={scale}")
    try:
        return cls(**cfg)
    except TypeError as e:  # unknown or missing kwarg
        raise ConfigError(f"[network] {name}: {e}") from e
```
`build_loss(cfg, *, scale)`, `build_dataset(cfg, *, seed)`, `build_model(cfg, runtime)`, `calculate_metric(...)` use the same table shape (`LOSSES`, `DATASETS`, `MODELS`, `METRICS`). `vgg_perceptual_loss` gets `scale: int` in its ctor (row `("vgg_perceptual_loss", "scale")`).

### Arch file after migration (e.g. dat_arch.py)
```python
# removed: from neosr.utils.registry import ARCH_REGISTRY
# removed: from neosr.archs.arch_util import net_opt  (keep other helpers)
# removed: upscale, __ = net_opt()
class dat(nn.Module):
    def __init__(self, ..., upscale: int = 4, ...):   # was `upscale=upscale`
        ...
def dat_s(**kwargs):                                  # decorator removed, body unchanged
    return dat(in_chans=3, ..., **kwargs)
```

### arch_util.py
```python
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        super().__init__()
        self.drop_prob, self.scale_by_keep = drop_prob, scale_by_keep
    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
# DySample.forward: coords_h = torch.arange(H, device=x.device, dtype=x.dtype) + 0.5 ; normalizer = x.new_tensor([W, H]).view(1, 2, 1, 1, 1)
# net_opt() deleted
```

### neosr/config.py
```python
class ConfigError(ValueError):
    def __init__(self, msg: str, *, known: Iterable[str] = ()):  # adds "did you mean" from difflib
        ...

# framework-owned sections: known keys -> (type, default | REQUIRED)
SCHEMA = {
    "": {"name": (str, REQUIRED), "model_type": (str, REQUIRED), "scale": (int, REQUIRED),
         "num_gpu": (int | str, "auto"), "manual_seed": (int, None), "use_amp": (bool, False),
         "bfloat16": (bool, False), "fast_matmul": (bool, False), "compile": (bool, False),
         "find_unused_parameters": (bool, False), "network_g": (dict, REQUIRED), "network_d": (dict, None),
         "datasets": (dict, REQUIRED), "train": (dict, None), "val": (dict, None), "logger": (dict, None),
         "path": (dict, {}), "degradations": (dict, None), "dist_params": (dict, None), "input_size": (dict, None)},
    "datasets.*": {"name", "type", "dataroot_gt", "dataroot_lq", "meta_info", "patch_size", "batch_size",
                   "accumulate", "num_worker_per_gpu", "pin_memory", "persistent_workers", "dataset_enlarge_ratio",
                   "augmentation", "aug_prob", "use_hflip", "use_rot", "color", "mean", "std", "queue_size", "io_backend"},
    "train": {"total_iter", "warmup_iter", "ema", "grad_clip", "clamp", "sam", "sam_init", "eco", "eco_init", "eco_iters",
              "eco_schedule", "wavelet_guided", "wavelet_init", "match_lq_colors", "enable_qat", ...},  # + *_opt / optim_* / scheduler as opaque tables
    "logger": {"print_freq", "save_checkpoint_freq", "use_tb_logger", "save_tb_img", "wandb"},
    "val": {"val_freq", "save_img", "save_lq", "suffix", "tile", "pbar", "metrics"},
    "path": {"pretrain_network_g", "pretrain_network_d", "param_key_g", "param_key_d", "strict_load_g", "strict_load_d",
             "resume_state", "ignore_resume_networks", "experiments_root", "results_root", "print_network"},
    "degradations": {... the 30 keys used by options/*.toml ...},
}
OPAQUE = re.compile(r"^(network_[gd]|train\.(\w+_opt|optim_[gd]|scheduler)|val\.metrics\.\w+)$")
MOVED = {("logger", "total_iter"): ("train", "total_iter")}   # accepted with a one-line deprecation log

def load_config(path: Path, *, is_train: bool, root: Path, debug: bool = False) -> dict[str, Any]:
    cfg = tomllib.loads(Path(path).read_text("utf-8"))   # TOMLDecodeError -> ConfigError(f"{path}: {e}")
    _migrate(cfg); _validate(cfg, is_train); _derive(cfg, root, is_train, debug)
    return cfg
# _derive: phase/scale into each dataset, [degradations] merged into datasets.train (one namespace),
# expanduser on paths, experiments/results roots, debug overrides (options.py:225-276 moved here unchanged).
```

### neosr/runtime.py
```python
@dataclass(frozen=True)
class Runtime:
    device: torch.device; rank: int; world_size: int; seed: int; deterministic: bool

def setup_runtime(cfg: dict, *, launcher: str = "none") -> Runtime:
    if launcher != "none": init_dist(launcher, **cfg.get("dist_params", {}))   # also mp start method
    rank, world = get_dist_info()
    seed = cfg.get("manual_seed"); deterministic = seed is not None
    if seed is None: seed = random.randint(1024, 10000)
    seed_all(seed + rank)                      # random, numpy legacy, torch (cpu+cuda)
    _configure_backends(cfg, deterministic)    # cudnn.benchmark / deterministic / env vars / fast_matmul (options.py:198-212, train.py:342-346)
    device = torch.device("cuda", torch.cuda.current_device())
    cfg.update(dist=launcher != "none", rank=rank, world_size=world, num_gpu=..., manual_seed=seed, deterministic=deterministic)
    return Runtime(device, rank, world, seed, deterministic)
```

### neosr/data/__init__.py
```python
def worker_init_fn(_: int) -> None:
    info = torch.utils.data.get_worker_info()        # torch already seeded random/torch/np legacy from base_seed + worker_id
    info.dataset.rng = np.random.default_rng(info.seed)

def build_dataset(cfg, *, seed: int) -> Dataset:
    module, _ = DATASETS[cfg["type"]]
    ds = getattr(importlib.import_module(f"neosr.data.{module}"), cfg["type"])(cfg)
    ds.rng = np.random.default_rng(seed)             # used as-is when num_workers == 0
    return ds
# build_dataloader: one argument dict for dist and non-dist (fixes the UnboundLocalError), worker_init_fn=worker_init_fn
# EnlargedSampler: g = torch.Generator(); g.manual_seed(self.epoch + seed); torch.randperm(n, generator=g)  (CPU)
```

### train.py
```python
def main() -> int:
    args = train_args()
    try:
        cfg = load_config(args.opt, is_train=True, root=ROOT, debug=args.debug)
        rt = setup_runtime(cfg, launcher=args.launcher)
        train(cfg, rt, args)
    except ConfigError as e:
        print(f"{tc.red}config error: {e}{tc.end}", file=sys.stderr); return 2
    except KeyboardInterrupt:
        ...save latest (train() installs the model in a nonlocal so this can save) ; return 130
    return 0

if __name__ == "__main__":
    mp.set_start_method("spawn"); sys.exit(main())
```

## sys.exit recipe (37 sites)
| site | replacement |
|---|---|
| image.py:310-433 (14 config checks) | moved to `config.validate_train(cfg)` (needs only cfg) raising ConfigError before any GPU allocation; the `patch_size % 4` check (:336, never raised) becomes a real error |
| base.py:199, :227 | `ConfigError(f"unsupported optimizer/scheduler {t}")` |
| base.py:377, :499 | after 3 retries `raise OSError(f"cannot save {path}")` (caller decides) |
| base.py:407 | `FileNotFoundError(load_path)` |
| base.py:445 | drop the bare except; let `load_state_dict` raise `RuntimeError` |
| data/otf_dataset.py:58 | ConfigError in ctor |
| data/otf_dataset.py:166, paired_dataset.py:129 | `raise OSError(f"could not read {path} after 3 retries")` — DataLoader propagates worker exceptions; sys.exit in a worker kills only the worker and the loader hangs |
| data/transforms.py:122,127 | ValueError |
| utils/lmdb_util.py:106,199 | ValueError / FileExistsError |
| options.py:38 | unreachable (after raise); module deleted |
| train.py:183 | ConfigError (unknown dataset phase) — moved into validation |
| train.py:474 | unreachable, deleted |
| train.py:488, :511 | ConfigError in validation (bf16 without amp; bf16 on pre-Ampere is a runtime check -> RuntimeError in setup_runtime) |
| train.py:584 | deleted (throughput kill) |
| train.py:597 | `raise RuntimeError("disk space")` after save |
| train.py:636, :642, test.py:68 | handled by `main()` |

## Migration steps
S0 (serial, one agent, new files only): `neosr/config.py`, `neosr/runtime.py`, `neosr/cli.py`, package tables + new `build_*` signatures, `worker_init_fn`, tests: `tests/test_imports.py` (import every module with `sys.argv=['x']` and no `-opt`), `tests/test_config.py` (load+validate all 129 option files; expect exactly the known-invalid ones to fail), `tests/test_archs.py` (parametrized over ARCHS: build at scale 2 and 4, CUDA forward 1x3x32x32 in train and eval, assert shape, no default device set), `tests/test_losses.py` (build each loss, CUDA forward on random pair, finite, no default device).
S1 (parallel, disjoint files; each sub-step verifiable alone):
- S1a 46 arch files, recipe R1 (one file = one unit; any number of agents).
- S1b 9 remaining arch files: decorator/import removal only.
- S1c arch_util.py (DropPath, DySample device, delete net_opt).
- S1d losses/*: decorator removal; vgg_perceptual_loss `scale` ctor arg; device fixes where tests/test_losses.py fails.
- S1e data/*: datasets `self.rng`, degradations `rng=` parameter, augmentations `rng` parameter, loader/sampler fixes, delete utils/rng.py.
- S1f models/base.py, image.py, otf.py: `device` from Runtime, sys.exit -> raise, `self.rng`, build_* new signatures.
- S1g train.py, test.py, convert.py: `main()` as above; remove `set_default_device`, the `common/` sys.path hack (train.py:673-679), the throughput kill.
S2 (serial): full test suite; 30-iteration `--debug` smoke runs on one paired, one otf and one GAN config; save -> resume -> 10 more iterations; `python convert.py` of the saved model.
S3 (serial): delete `neosr/utils/options.py`, `neosr/utils/rng.py`, `neosr/utils/registry.py`, star imports in `neosr/__init__.py`, `soap_sf.py:9-20` dead text.

R1 per arch file: (1) delete `net_opt` from the import line (delete the line if it becomes empty) and the `from neosr.utils.registry import ARCH_REGISTRY` line; (2) delete the module-level `upscale, __ = net_opt()` / `upscale, training = net_opt()` line; (3) in the registered class ctor replace the default `= upscale` by `= 4` (keep the parameter name); (4) delete every `@ARCH_REGISTRY.register()` line; (5) confirm the file's registered names and scale kwarg match the `ARCHS` rows; (6) verify: `python -I -c "import sys; sys.argv=['x']; import neosr.archs.<file>"` and `pytest tests/test_archs.py -k <names>`; if the forward fails with a device mismatch, add `device=x.device` (or `x.new_*`) to the flagged factory call only.

## What I would not do
- Typed config structs (msgspec/pydantic as in traiNNer-redux `redux_options.py`): rewrites every `opt[...]` read, adds a dependency; the dict + schema gives the same unknown-key safety.
- Normalize all 46 archs to a `scale` kwarg: edits vendored ctor bodies for no contract gain over the table.
- A `Config` singleton (traiNNer `Config.get_scale()`): same ambient coupling as `net_opt()`, just lazier.
- Keep `set_default_device` scoped with `with torch.device(...)`: still ambient, still the TorchFunctionMode cost, still wrong under DataParallel (replicas on cuda:1.. would allocate on cuda:0).
- Blindly add `device=` to all 335 device-less factory calls; only the probe-flagged ones.
- try/except auto-discovery of archs: hides broken modules.
