# T2 — Model/trainer structure, validation, checkpoint/resume (merged design)

Inputs: Opus position, Fable proposal (`design/t2_design.md`), Codex proposal; aligned with the T1 (loss pipeline / training step), T3 (config, RNG, entry points) and T4 (feature triage, perf) drafts in the same directory. Conflicts settled by reading `/home/korvin/apps/neosr-dev` (read-only; paths below are relative to it). New probe: `design/t2probe/tile_probe.py` (tiling exactness). Fable's probes on torch 2.9.1 are reused where cited as [F-probe].

## 0. Design in ten lines

1. Three model files: `base.py` (generic plumbing), `image.py` (the SR trainer), `otf.py` (one `feed_data` override). Registered names `image`/`otf` unchanged. `default.py`, `simple_otf.py`, `eqr_otf.py`, `pancam_otf.py`, `image.py.helaman` are deleted.
2. OTF synthesis (degradation + pair pool) moves to `neosr/data/otf_degradation.py`. It is testable without building a trainer, and validation never touches it.
3. Each net has one canonical eager module (`self.net_g`, `self.net_d`) that is used for loading, saving, EMA and validation. It also has one execution wrapper (`self.net_g_fwd = compile(DDP(net_g))` or the same object) that is used only for the training forward.
4. Construction order is build → pretrained → EMA → T1 phases (optimizer, scaler, schedulers) → resume → wrap. Resume therefore happens inside `__init__`, loads into bare modules, and runs before DDP's broadcast and before compile.
5. There are two artifacts. `models/net_{g,d}_<update>.pth` is a weights-only `{"params": sd}` export (G = EMA if enabled, else the eval weights; D = live). `training_states/<update>.state` (format 2) is self-contained: live G/D, EMA, per-phase `{optimizer, scaler, schedulers}`, `update`, `epoch`, RNG and best metrics. Resume never reads a `.pth`.
6. One counter is user-visible and persisted: `update` (completed optimizer windows). Saves happen only at update boundaries, and Ctrl-C stops at the next boundary.
7. One context manager, `eval_weights()`, serves validation and export. It takes the EMA module as is. Without EMA it uses the live module in its schedule-free `x` representation via snapshot → `optimizer.eval()` → use → exact restore. `.state` is always in train representation.
8. Tiled validation uses fixed-size tiles snapped inside the image with a context halo and core crop. The result equals full inference to fp round-off when the receptive field is ≤ halo (probe). There is no exterior padding and no CPU shuttle.
9. Atomic IO: temp file in the same directory → fsync → `os.replace` → directory fsync, with retry only on `OSError`. `.state` is written before the exports. A failed resume raises. It never starts fresh and never archives the experiment folder.
10. A single reader, `read_weights()`, is shared by train/test/convert. An explicit `param_key` wins or errors. Otherwise the precedence is `params_ema` > `params-ema` > `params` > bare dict. `module.`/`_orig_mod.` prefixes are stripped and `n_averaged` is dropped. Legacy v1 `.state` + `.pth` pairs resume through one adapter.

## 1. Decisions and conflict resolution

| # | Topic | Decision | Chosen over | Evidence / reason |
|---|---|---|---|---|
| D1 | Trainer classes | `base` + `image` + `otf(image)`, keep `MODEL_REGISTRY` scan | Codex: one `ImageTrainer` + `PairedPreparation/OTFPreparation` + explicit factory + 8 model files | Only `image` (87) and `otf` (42) appear in options/. The single difference between them is how a training batch is made (`otf.py:150-280` vs `image.py:444-460`), which one override expresses. A preparation protocol plus factory adds a type and a file to express the same thing. Archs use the same registry pattern, so keeping it is consistent. |
| D2 | Legacy trainers | Delete the 4 files + `image.py.helaman`. An unknown `model_type` gets the registry's error listing `image, otf` | Codex: dedicated "feature removed" errors | 0 configs (T4 table). The generic message already names the valid types without needing a special-case list. |
| D3 | OTF location | `neosr/data/otf_degradation.py`: `Degrader` (two-stage RealESRGAN chain, owns `DiffJPEG`, reads one merged config) and `PairPool` (T4 §3.9 random-slot swap) | Opus: keep the code in `otf.py` | Testable with a seed and a tensor, no nets/optimizers/losses needed. Today it is spread across root and dataset keys (`otf.py:146,153,207,229`; T3 merges `[degradations]` into `datasets.train` once). |
| D4 | Canonical vs execution module | `self.net_g` = bare eager. `self.net_g_fwd` = `compile(DDP(net_g))` / `compile(net_g)` / `net_g`. Validation is eager. | Fable: `self.net_g` = wrapper + `bare()` unwrapping everywhere | T4 P7 verified that compile-before-load breaks keys (`base.py:163` before `load_network`) and leaks `_orig_mod.` into exports (`base.py:347`). Keeping two references removes every unwrap call site. `bare()` survives only inside `read_weights` prefix stripping for old files. |
| D5 | DataParallel | Removed. `num_gpu > 1` without a torchrun launch raises "use torchrun (DDP)". | Codex: keep DP | 0 configs set `num_gpu` or `compile`. DP + compile is unsupported upstream, and DP is a third wrapping path to test for no user. |
| D6 | EMA class | T1 `Ema` (`neosr/models/ema.py`): deepcopy of the bare module after the topology is final, `update(finite)` = masked `_foreach_lerp_` + buffer copy | Fable: `torch.optim.swa_utils.AveragedModel` | `AveragedModel.update_parameters` evaluates `bool(self.n_averaged == 0)` every update (`torch/optim/swa_utils.py:262`), a host sync. T1 needs a non-finite mask without a sync (T1 §2.3). `get_ema_multi_avg_fn` ignores the count (`swa_utils.py:45`), so `n_averaged` carries no information. Dropping it also settles the Fable/Codex legacy debate (`n_averaged = iter` vs `0`). |
| D7 | D EMA | Deleted. `train.ema` applies to G only. | — | All three proposals and T4 agree. Its only effect today is resuming D from its average (`image.py:998-999`). |
| D8 | Export layout | `net_g_<u>.pth = {"params": sd}` only, with no `params_ema` and no `meta`. `net_d_<u>.pth = {"params": live D}` is kept. | Opus "+params_ema?", Fable `meta` + no D export, Codex optional D | `"params"` is what every existing experiment and loader reads (`convert.py:95-104`, `base.py:414-419`). A second tensor set doubles size and resume does not need it. `meta` can be mis-picked by "first value" loaders and `.state` already holds the build info. D is kept because 85 templates carry `#pretrain_network_d`: a new GAN run that fine-tunes from a previous run needs a D file, and resume is not the right tool when the config changed. |
| D9 | `.state` content | Self-contained v2. Optimizer/scaler/schedulers are grouped per T1 `Phase.state_dict()`. | Codex: separate `config`, `compatibility`, `progress(7 fields)`, `rank_state`, `provenance`, `latest.json`; Fable: separate `build` block + `micro` | Each object serializes itself, so the trainer loops over `self.phases`/`self.nets` without per-type code (the owner's "no if-chains" rule applied to checkpoints). Compatibility comes from what actually breaks a load: optimizer type and ordered parameter names (Codex's point; counts alone do not prove the mapping). Net shapes are checked by `load_state_dict(strict=True)`. Precision, EMA and scheduler changes are tolerated with a warning (§6.4). |
| D10 | Counters | Persist `update` + `epoch`. The window position `micro` restarts at 0 because saves happen only at boundaries. | Codex: `micro/window/updates_g/updates_d/offset/next_batch` | AMP-skipped updates are counted as windows. Per-optimizer step counts already live in the optimizer state (`step`), and T1 logs `skipped_g/d`. Resuming with a different `accumulate` is then automatically consistent. |
| D11 | RNG | Rank-0 main-process RNG (python, legacy numpy, `neosr.utils.rng` PCG64, torch CPU, CUDA) is saved and restored by `model.restore_rng()`, which train.py calls immediately before the loop. | Fable: restore at the end of `resume()`; Codex: per-rank gather | Between `build_model` and the loop, train.py consumes RNG (MACs profiling `torch.randn`, `train.py:398-402`; prefetcher construction). Restoring inside `__init__` would be consumed before use. Per-rank gather is a collective for a single-GPU user base. Other ranks reseed as at startup. |
| D12 | OTF pool persistence | Not persisted; it refills after resume. | Codex: serialize the pool | 180 × HR patch × 2 tensors ≈ 150 MB per state (T4: "2×141 MB gathers"), for an effect on the first `queue_size/batch` batches only. Exact data-order resume is out of scope (§10, owner question). |
| D13 | Tiling | Fixed-size tiles snapped inside the image, context halo `val.tile_overlap` (default 16 LR px), keep each tile's core, stitch on the output device | Fable: ragged clipped tiles + CPU stitch option; T4: overlap + linear blend | Probe: equals full inference within 1e-16 (fp64) when halo ≥ receptive field, and 1e-3 off when halo < RF. Fixed tile shape keeps window-size archs and cudnn autotune happy (Fable's ragged tiles do not). Crop gives exactness for finite RF where blending only smooths. |
| D14 | QAT | No QAT in the trainer (T4: `enable_qat` → removed-key error; `scripts/aether_qat.py`). If the owner keeps it, the hook slot is §3 step 3 and is 4 lines. | Fable: structural hook in `__init__`; Codex: `qat.py` adapter | QAT is broken in three places today (T4 row "aether"). Keeping a trainer hook for one arch is the kind of special case the owner wants gone. The slot is specified so the decision is reversible. |
| D15 | `ignore_resume_networks` | Removed key (T3 `REMOVED` table) | Codex: reinterpret as partial continuation | 0 configs. Use `pretrain_network_*` for a new run. |
| D16 | Atomic write | `tempfile.NamedTemporaryFile(dir=..., delete=False)` → fsync → `os.replace` → dir fsync (posix). 3 attempts on `OSError` only, then raise. Any other exception unlinks the temp file and re-raises. | Fable per-attempt names; Codex no retry | The retry is for Windows file locks (configs use `C:\` paths; `os.replace` fails while a viewer holds the target). Today's loop reports success as failure and `sys.exit(1)`s (`base.py:361-377`). |
| D17 | Auto-resume | Picks the newest `<int>.state`. A load failure raises with the path. An explicit `resume_state` must exist. | Fable: validate newest-first and fall back | Atomic writes leave no partial `.state`, so fallback only hides disk faults. Loading 300+ MB twice to probe is not free. Today a failure becomes a fresh start that archives the folder (`train.py:235-238` → `make_exp_dirs` at 350). |

## 2. File layout

```
neosr/models/
  __init__.py         unchanged: registry scan + build_model(opt)
  base.py   ~180 l    class base: device, wrap, OPTIMIZERS table + build_optimizer (T4 §3.10), build_schedulers,
                      save / resume / restore_rng / validation driver / best metrics / get_current_log
  image.py  ~330 l    class image(base): nets, EMA, phases (T1), feed_data, optimize_parameters (T1 body),
                      eval_weights, infer, validate, export_state_dicts
  otf.py     ~35 l    class otf(image): Degrader + PairPool, feed_data override
  checkpoint.py ~170 l  pure functions: STATE_FORMAT, strip_prefixes, cpu_copy, read_weights, load_weights,
                      atomic_save, latest_state, rng_state/set_rng_state, upgrade_v1
  phase.py, ema.py, eco.py   (T1; T2 adds Phase.state_dict/load_state_dict, §7.2)
  tiling.py  ~30 l    infer_tiled(net, lq, scale, tile, halo) — pure, shared by validation and test.py
neosr/data/
  otf_degradation.py  Degrader, PairPool (lifted from models/otf.py:37-280)
```

Deleted: `neosr/models/{default,simple_otf,eqr_otf,pancam_otf}.py`, `image.py.helaman`, `neosr/utils/misc.py:check_resume` (132-167) and its export (`neosr/utils/__init__.py`), `train.py:load_resume_state` (192-239), and in base.py `save_network`, `load_network`, `_print_different_keys_loading`, `save_training_state`, `resume_training`, `get_bare_model`, `model_to_device`, `update_learning_rate`/`_set_lr`/`_get_init_lr` (T1: schedulers live in phases, warmup goes away), `update_loss_weights`/`get_current_gan_weight`, and the `dist_validation`/`nondist_validation` pair. The legacy datasets that lose their consumers are T4 lane A.

## 3. Construction order (`image.__init__`)

```
1. self.net_g = build_network(opt["network_g"]).to(device, memory_format=mf)        # bare, eager
2. if pretrain_network_g and not resuming: load_weights(self.net_g, read_weights(path, param_key_g), strict_load_g)
3. [QAT slot, only if the owner keeps QAT: net_g.prepare_qat(cfg); calibrate; one dummy forward — topology final]
   if not is_train: wrap and return                                                  # test.py
4. self.ema = Ema(self.net_g, train.ema) if train.ema > 0 else None                  # after the last structural change
5. same 1-2 for net_d; self.phases = {"net_g": Phase(...), "net_d": Phase(...)}      # T1; optimizers from trainable(net)
   self.nets = {"net_g": self.net_g, "net_d": self.net_d}  (net_d entry absent without D); losses (T1)
6. self.update, self.epoch = 0, 0;  if resuming: self.resume(path)                  # into bare modules
7. self.net_g_fwd = self.wrap(self.net_g); self.net_d_fwd = self.wrap(self.net_d)   # DDP then compile, last
```

Invariant: anything that captures parameter identity or buffer shape (EMA `src/dst` lists, optimizer param lists) is created after the last structural change and before wrapping. Resume loads into the same tensors, and compiled graphs and DDP read them [F-probe: `load_state_dict` into the module behind `_orig_mod` is seen by the compiled graph]. Under DDP, resuming before wrapping means DDP's constructor broadcast starts from the restored rank-0 weights.

`trainable(net) -> (names, params)` = `[(n, p) for n, p in net.named_parameters() if p.requires_grad]`. It is the only source for both the optimizer's param list and the names stored in the state, so their order cannot diverge.

Pretrained loading is skipped when resuming, because the state carries the weights. This replaces `check_resume` rewriting `pretrain_network_*` to the `.pth` files (`misc.py:150-159`). Without that, live weights resume from the EMA export (`image.py:992-1003`).

## 4. Counters, boundary, stop

| Name | Meaning | Persisted |
|---|---|---|
| `self.update` | completed windows (`accumulate` micro-batches each); an fp16-skipped step still counts, so schedules keep their length | `state["update"]`, file names, log `iter`, `val_freq`/`save_checkpoint_freq`/`total_iter`, the `it` T1 passes to losses (`wavelet_init`, ECO) |
| `self.micro` | micro-batches in the current run, used only for `micro % accumulate` | no (always 0 at a boundary) |
| `self.epoch` | sampler epoch at save | `state["epoch"]` |

T1's `optimize_parameters() -> bool` increments `self.update` when it steps. train.py never recomputes accumulation. This fixes save-as-updates/resume-as-microbatches (`train.py:600` vs `454`) and the second division (`logger.py:143,151`).

Graceful stop: train.py installs a SIGINT handler that sets a flag. The loop checks it after a completed update, then validates (if configured), saves `final` and exits 0. A second SIGINT restores the default handler and the last committed checkpoint stands. No save exists mid-window, between the G and D steps, or inside a loss.

Epoch budgeting: the loop is `while not done` with `epoch += 1` per sampler pass. `total_epochs` and the mixed-unit arithmetic (`train.py:148-166`: `num_iter_per_epoch` in updates, `total_iters` in micro-batches) disappear. A resume with `update >= total_iter` exits before touching data.

## 5. Validation and inference

```python
# image.py
@contextmanager
def eval_weights(self):
    """The generator to evaluate, in eval mode; restores everything exactly. Never entered inside inference_mode."""
    if self.ema is not None:                       # EMA module is permanently eval / requires_grad False
        yield self.ema.module; return
    net, phase = self.net_g, self.phases.get("net_g")
    swap = phase is not None and hasattr(phase.opt, "eval")          # schedule-free (adamw_sf.py:75-97)
    was_training = net.training
    snap = [p.detach().clone() for p in net.parameters()] if swap else None
    if swap: phase.opt.eval()                       # p <- x
    net.eval(); self._evaluating = True
    try:
        yield net
    finally:
        self._evaluating = False
        net.train(was_training)
        if swap:                                    # exact restore; eval()->train() is two lerps, not bit-inverse
            with torch.no_grad():
                torch._foreach_copy_(list(net.parameters()), snap)
            for g in phase.opt.param_groups: g["train_mode"] = True
```

The exact restore matters beyond accuracy. Under DDP only rank 0 validates. A lossy `eval()`/`train()` round trip would leave rank 0's weights different from the other ranks', and nothing would re-sync them.

```python
def validate(self, loader, update, tb_logger, save_img):          # rank 0, called by base.validation
    tile, halo = self.val_tile, self.val_tile_overlap
    with self.eval_weights() as net:                                # one swap per validation, not per image
        for data in loader:
            with torch.inference_mode():
                lq = data["lq"].to(self.device, non_blocking=True)
                sr = net(lq) if tile <= 0 else infer_tiled(net, lq, self.scale, tile, halo)
            self.metrics.update(sr, data.get("gt"), name=..., save_img=save_img)   # metrics topic; float() values
    results = self.metrics.compute()                                # mean over actual image count; empty loader rejected at startup
    self.update_best(loader.dataset.opt["name"], results, update)
```
Validation runs in fp32, as today (`image.py:896-897` has no autocast). The `is_train` flips (`image.py:851,980`; `otf.py:286-292`) are gone because validation never calls `feed_data`.

```python
# base.py
def validation(self, loader, update, tb_logger, save_img):
    dist = torch.distributed.is_initialized()
    if dist: torch.distributed.barrier()
    if self.opt["rank"] == 0:
        with torch.random.fork_rng(devices=[self.device] if self.device.type == "cuda" else []):
            self.validate(loader, update, tb_logger, save_img)    # loader workers/metrics cannot shift training RNG
    if dist: torch.distributed.barrier()
```

Tiling (`neosr/models/tiling.py`; verified by `design/t2probe/tile_probe.py`):
```python
def _starts(n, tile, halo):
    return [0] if n <= tile else [*range(0, n - tile, tile - 2 * halo), n - tile]

def _core(s, n, tile, halo):                       # LR span this tile writes
    return (s + halo if s > 0 else 0), (s + tile - halo if s + tile < n else n)

def infer_tiled(net, lq, scale, tile, halo):
    """Fixed tile x tile windows snapped inside the image (smaller only when the image is), each writing its
    core; context never leaves the image, so borders see what full inference sees."""
    _, _, h, w = lq.shape; out = None
    for y in _starts(h, tile, halo):
        y0, y1 = _core(y, h, tile, halo)
        for x in _starts(w, tile, halo):
            x0, x1 = _core(x, w, tile, halo)
            o = net(lq[..., y:y + tile, x:x + tile])
            if out is None:                         # channels/dtype from the first tile
                out = o.new_empty(o.shape[0], o.shape[1], h * scale, w * scale)
            out[..., y0 * scale:y1 * scale, x0 * scale:x1 * scale] = \
                o[..., (y0 - y) * scale:(y1 - y) * scale, (x0 - x) * scale:(x1 - x) * scale]
    return out
```
The construction-time check is `tile > 2 * tile_overlap`. Probe results (fp64, 6-conv net with RF = 6 LR px, 2×, 3→4 channels): halo 6/7/8 gives max diff ≤ 1.7e-16 against full inference for shapes 37×53, 64×64, 10×70, 100×33 and 5×5. Halo 3 gives 1.0e-3. The output stays on the device. One `.cpu()` happens in the metric/image path, with no per-tile `empty_cache` (T4 P9: +110 ms per call). `val.tile = -1` keeps meaning full-image inference. `val.tile_overlap` is a new optional key.

## 6. Checkpoint formats

### 6.1 Exports (user-facing, `models/`)

```python
net_g_<u>.pth = {"params": OrderedDict}   # bare keys, CPU copies, contiguous: EMA if ema>0 else eval_weights() (SF x / plain live)
net_d_<u>.pth = {"params": OrderedDict}   # live D (GAN configs only)
```
At the end of a run (normal or graceful stop), `net_g_latest.pth`/`net_d_latest.pth` are also written. That is today's final file name (`base.py:330`), and today the final `.state` is suppressed (`base.py:461`). Now the final `<u>.state` is written as well.

### 6.2 `training_states/<u>.state` (format 2)

```python
{
  "format": 2, "update": int, "epoch": int,
  "nets":   {"net_g": sd, "net_d": sd},          # LIVE bare weights; schedule-free: train (y) representation
  "ema":    sd | None,                            # Ema.module.state_dict(); None when train.ema <= 0
  "phases": {"net_g": {"optimizer": {"type": "adan_sf", "names": [str, ...], "state": optimizer_sd},
                       "scaler": GradScaler_sd,   # {} when disabled [F-probe]
                       "schedulers": [sched_sd, ...]},
             "net_d": {...}},
  "rng":    {"python": tuple, "numpy": {"key": uint32 Tensor, "pos": int, "has_gauss": int, "gauss": float},
             "neosr": rng.bit_generator.state, "torch": ByteTensor, "cuda": [ByteTensor]},   # rank 0
  "best":   {dataset: {metric: {"better": str, "val": float, "iter": int}}},
}
```
Rules:
- The state must round-trip through `torch.load(weights_only=True)`. [F-probe] covers python/torch/CUDA RNG, optimizer dicts and PCG64 state dicts (plain ints). Numpy arrays are stored as tensors, and numpy scalars are refused, hence `float()` for metric values (PSNR returns a numpy scalar, `calculate.py:65`).
- Tensors are stored as-is (device tensors included). `torch.save` is synchronous, so no snapshot is needed. Load uses `map_location="cpu"`, and `load_state_dict` copies or casts to the live device.
- Size compared with today: today = G(EMA) export + D(EMA) export + 2 optimizer states. New = G(EMA) export + D export + G live + G EMA + D live + 2 optimizer states, so +2 G and +1 D copies per checkpoint. The optional `logger.keep_states = N` (default 0 = keep all, as today) prunes older `.state` files after a successful write.

### 6.3 Pretrained reading (`read_weights`)

Accepted inputs: `{"params_ema"|"params-ema"|"params": sd}`, a bare `sd`, keys with any number of leading `module.`/`_orig_mod.`, with or without `n_averaged`. An explicit `param_key_*` must exist or the load raises with the top-level keys listed. No config sets `param_key_*` today (grep), so honoring it changes no existing run. `strict_load_* = true` (default) means exact names and shapes. With `false`, shape-mismatched keys are dropped with a warning, missing and unexpected keys are listed, and a load that matched zero tensors raises. `map_location="cpu"`. No `mmap`, because legacy non-zip files reject it (`torch/serialization.py:1537`, Fable).

### 6.4 Resume tolerance (what may change between runs)

| Changed in TOML | Behaviour |
|---|---|
| lr, loss weights, freqs, `total_iter`, datasets, `accumulate`, batch size | Allowed. Optimizer `param_groups` (lr) come from the state, as today. |
| network options that change shapes or names | `load_state_dict(strict=True)` raises with torch's message |
| optimizer `type` or the trainable parameter list | raises before loading that optimizer, naming the first differing parameter |
| `use_amp`/`bfloat16` (scaler enabled ↔ disabled) | Warning. A scaler with an empty or disabled saved state starts fresh, and a saved scaler state is ignored when the scaler is now disabled. |
| `ema` enabled/disabled | Warning. A missing saved EMA is initialized from live G, and a saved EMA is ignored when EMA is now disabled. |
| number of schedulers | Warning; the schedulers start fresh. |
| world size | Warning; ranks ≠ 0 reseed (RNG is exact for rank 0 only). |

Any raise happens inside `__init__` and aborts the run, so partial mutation of the in-memory model is irrelevant. That is why no separate "validate everything first" pass is needed.

### 6.5 Legacy v1 resume

A legacy state is recognised by a missing `"format"` key: `{"epoch", "iter", "optimizers": [g, d], "schedulers": [...]}` from `base.py:459-470`, plus `models/net_g_<iter>.pth` and `net_d_<iter>.pth`. `upgrade_v1` converts it as follows:
- `update = iter`. The file name and the stored value were already windows (`train.py:600`); `train.py:454` mis-read it as micro-batches.
- `nets` come from the two `.pth` files via `read_weights`, which raises if they are missing. They may hold EMA weights rather than live ones, and the warning says so.
- `ema` = the same G weights.
- `phases` are zipped from the lists in order `("net_g", "net_d")` (the append order in `image.py:383` and below), with `names = None`. The name check is skipped, and torch's own group/size check in `Optimizer.load_state_dict` remains.
- Scalers are empty, which triggers the fresh-scale warning. `rng` and `best` are absent.

One warning lists what was reset. The fixture is `experiments/train_esrgan/training_states/3236.state` + `models/net_{g,d}_3236.pth` ([F-probe]: loads with `weights_only=True`).

## 7. Skeletons

### 7.1 `neosr/models/checkpoint.py`

```python
STATE_FORMAT = 2
_PREFIXES = ("module.", "_orig_mod.")

def strip_prefixes(sd):
    out = OrderedDict()
    for k, v in sd.items():
        while k.startswith(_PREFIXES):
            k = k.split(".", 1)[1]
        if k == "n_averaged":
            continue
        if k in out:
            raise KeyError(f"prefix stripping collides on {k!r}")
        out[k] = v
    return out

def cpu_copy(sd):                                  # independent even when training on CPU (Codex)
    return OrderedDict((k, v.detach().to("cpu", memory_format=torch.contiguous_format, copy=True)) for k, v in sd.items())

def read_weights(path, param_key=None):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if param_key is not None:
        if param_key not in obj:
            raise KeyError(f"{path}: param_key {param_key!r} not found; keys: {list(obj)[:10]}")
        return strip_prefixes(obj[param_key])
    for key in ("params_ema", "params-ema", "params"):
        if key in obj:
            return strip_prefixes(obj[key])
    if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return strip_prefixes(obj)
    raise KeyError(f"{path}: no parameter mapping; keys: {list(obj)[:10]}")

def load_weights(net, sd, strict, log):
    if not strict:
        own = net.state_dict()
        for k in [k for k in sd if k in own and own[k].shape != sd[k].shape]:
            log.warning(f"shape mismatch, skipped {k}: {tuple(sd[k].shape)} -> {tuple(own[k].shape)}"); del sd[k]
        if not sd.keys() & own.keys():
            raise RuntimeError("pretrained file matches no tensor of the network")
    res = net.load_state_dict(sd, strict=strict)
    for k in res.missing_keys: log.warning(f"missing: {k}")
    for k in res.unexpected_keys: log.warning(f"unexpected: {k}")

def atomic_save(obj, path, attempts=3):
    for attempt in range(1, attempts + 1):
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as f:
                tmp = Path(f.name)
                torch.save(obj, f); f.flush(); os.fsync(f.fileno())
            os.replace(tmp, path); tmp = None
            if os.name == "posix":
                fd = os.open(path.parent, os.O_RDONLY)
                try: os.fsync(fd)
                finally: os.close(fd)
            return
        except OSError:
            if attempt == attempts: raise
            time.sleep(1)
        finally:
            if tmp is not None: tmp.unlink(missing_ok=True)

def latest_state(d):
    return max((p for p in Path(d).glob("*.state") if p.stem.isdigit()), key=lambda p: int(p.stem), default=None)

def rng_state():
    name, key, pos, has_gauss, gauss = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": {"key": torch.from_numpy(key.astype(np.int64)), "pos": pos, "has_gauss": has_gauss, "gauss": gauss},
            "neosr": neosr_rng.bit_generator.state}          # T3: one process-wide Generator, reseeded in place

def set_rng_state(s): ...      # exact inverse
def upgrade_v1(state, models_dir, log): ...   # §6.5
```

### 7.2 Phase serialization (T2 addition to the T1 `Phase`)

```python
class Phase:                                   # T1 owns step/backward; T2 adds:
    # set in __init__: self.opt_type (TOML type string), self.names (from trainable(net))
    def state_dict(self):
        return {"optimizer": {"type": self.opt_type, "names": self.names, "state": self.opt.state_dict()},
                "scaler": self.scaler.state_dict(),
                "schedulers": [s.state_dict() for s in self.schedulers]}

    def load_state_dict(self, sd, log):
        o = sd["optimizer"]
        if o["type"] != self.opt_type:
            raise RuntimeError(f"optimizer changed: {o['type']} -> {self.opt_type}")
        if o["names"] is not None and o["names"] != self.names:
            first = next((a, b) for a, b in zip_longest(o["names"], self.names) if a != b)
            raise RuntimeError(f"trainable parameters changed at {first}")
        self.opt.load_state_dict(o["state"])                      # SF: restores train_mode=True groups
        if self.scaler.is_enabled():
            if sd["scaler"]: self.scaler.load_state_dict(sd["scaler"])
            else: log.warning("fp16 GradScaler starts fresh")
        if len(sd["schedulers"]) == len(self.schedulers):
            for s, ssd in zip(self.schedulers, sd["schedulers"]): s.load_state_dict(ssd)
        else: log.warning("scheduler configuration changed; schedulers start fresh")
```

### 7.3 `base.py`: save / resume / restore_rng

```python
def save(self, epoch, final=False):
    assert self.micro % self.accumulate == 0 and not self._evaluating, "save only at an update boundary"
    if self.opt["rank"] != 0: return
    u, models = self.update, Path(self.opt["path"]["models"])
    atomic_save(self.training_state(epoch), Path(self.opt["path"]["training_states"]) / f"{u}.state")  # authoritative first
    for name, sd in self.export_state_dicts().items():               # {"net_g": ..., "net_d": ...}
        payload = {"params": sd}
        atomic_save(payload, models / f"{name}_{u}.pth")
        if final: atomic_save(payload, models / f"{name}_latest.pth")
    self._prune_states()

def training_state(self, epoch):
    return {"format": STATE_FORMAT, "update": self.update, "epoch": epoch,
            "nets": {n: m.state_dict() for n, m in self.nets.items()},
            "ema": self.ema.module.state_dict() if self.ema is not None else None,
            "phases": {n: p.state_dict() for n, p in self.phases.items()},
            "rng": rng_state(), "best": self.best_metric_results}

def resume(self, path):
    log = get_root_logger()
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "format" not in state:
        state = upgrade_v1(state, Path(self.opt["path"]["models"]), log)
    elif state["format"] != STATE_FORMAT:
        raise RuntimeError(f"{path}: unsupported state format {state['format']}")
    for n, net in self.nets.items():
        net.load_state_dict(state["nets"][n])                           # strict
    if self.ema is not None:
        src = state["ema"]
        if src is None: log.warning("EMA enabled now but absent in the state: initialised from live G")
        self.ema.module.load_state_dict(src if src is not None else self.net_g.state_dict())
    for n, phase in self.phases.items():
        phase.load_state_dict(state["phases"][n], log)
    self.update, self.epoch = state["update"], state["epoch"]
    self.best_metric_results = state.get("best") or {}
    self._pending_rng = state.get("rng")
    log.info(f"resumed {path.name}: update {self.update}, epoch {self.epoch}; weights, EMA, optimizers, "
             "schedulers, scalers, counters exact; data order, worker augmentation and OTF pool restart")

def restore_rng(self):                       # train.py: last call before the loop
    if self._pending_rng is not None and self.opt["rank"] == 0:
        set_rng_state(self._pending_rng)
    self._pending_rng = None
```

### 7.4 `image.py`: export and wrapping

```python
def export_state_dicts(self):
    with self.eval_weights() as net:                         # EMA, SF-x or live; restored on exit
        out = {"net_g": cpu_copy(net.state_dict())}          # copy taken before the restore
    if self.net_d is not None:
        out["net_d"] = cpu_copy(self.net_d.state_dict())
    return out

def wrap(self, net):                                         # base.py
    if net is None: return None
    if self.opt["dist"]:
        net = DistributedDataParallel(net, device_ids=[self.device.index])
    if self.opt.get("compile"):
        net = torch.compile(net, mode=self.opt.get("compile_mode", "default"), dynamic=False)   # T4 P7
    return net
```

### 7.5 `otf.py`

```python
@MODEL_REGISTRY.register()
class otf(image):
    def __init__(self, opt):
        super().__init__(opt)
        cfg = opt["datasets"]["train"]                 # T3: [degradations] already merged here
        self.degrader = Degrader(cfg, scale=self.scale, device=self.device)   # validates every key once
        self.pool = PairPool(cfg.get("queue_size", 180), cfg["batch_size"], device=self.device)

    @torch.no_grad()
    def feed_data(self, data):
        gt = data["gt"].to(self.device, non_blocking=True)
        lq = self.degrader(gt, data["kernel1"], data["kernel2"], data["sinc_kernel"])
        gt, lq = paired_random_crop([gt], lq, self.patch_size, self.scale)  # patch_size = LR size (otf.py:252-255)
        self._set_batch(*self.pool(lq, gt))           # image._set_batch: memory format + augmentation
```
`image.feed_data(data)` = `self._set_batch(lq.to(device), gt.to(device))`. `_set_batch` applies `apply_augment` (today `image.py:449-460` / `otf.py:268-280`). Because of `no_grad`, the batch never becomes an inference tensor that later enters autograd (Codex).

### 7.6 train.py loop contract (T3 owns the file; this is the T2 interface)

```python
resume = latest_state(paths.training_states) if opt["auto_resume"] else opt["path"].get("resume_state")
opt["path"]["resume_state"] = resume                  # explicit path must exist (Path.is_file or raise)
if resume is None: make_exp_dirs(opt)                  # never archive a folder being resumed
model = build_model(opt)                               # resume happened inside
stop = install_stop_flag()                             # 1st Ctrl-C: stop at next boundary; 2nd: default handler
done = model.update >= total_updates
epoch = model.epoch
model.restore_rng()
while not done:
    sampler.set_epoch(epoch); prefetcher.reset()
    while (batch := prefetcher.next()) is not None:
        model.feed_data(batch)
        if not model.optimize_parameters(): continue   # inside a window
        u = model.update
        done = u >= total_updates or stop.is_set()
        if u % print_freq == 0: msg_logger({"epoch": epoch, "iter": u, "lrs": model.current_lr(), **model.get_current_log()})
        val_due, save_due = val_loaders and (u % val_freq == 0 or done), u % save_freq == 0 or done
        if val_due or save_due: model.check_finite()   # T1: raises before a NaN reaches disk
        if val_due:
            for loader in val_loaders: model.validation(loader, u, tb_logger, save_img)
        if save_due: model.save(epoch, final=done)     # validate first: saved "best" includes this update
        if done: break
    epoch += 1
```
Other exceptions propagate with their traceback (no `sys.exit(1)` wrapper, `train.py:637-642`). `test.py:55` keeps calling `model.validation(loader, current_iter=..., tb_logger=..., save_img=...)`, so the keyword names stay.

## 8. Migration (letters in one row = disjoint files, parallel-safe)

| Step | Files | Work | Verification |
|---|---|---|---|
| 1a | `neosr/models/checkpoint.py` (new) | §7.1 | CPU pytest. Prefix stripping covers `_orig_mod.module.x` and a collision raises. Key precedence holds and an explicit missing key raises. Relaxed load with zero matches raises. `atomic_save` leaves no `.tmp`: 2 injected `OSError` then success, 3 → raise, an injected `RuntimeError` in `torch.save` → no retry, temp removed. `latest_state` ignores `.tmp` and non-digit stems. `read_weights` works on `experiments/train_esrgan/models/net_g_2000.pth`. `rng_state()` round-trips under `weights_only=True`. |
| 1b | `neosr/data/otf_degradation.py` (new) | Degrader, PairPool (T4 §3.9 swap) | Seeded: `lq` `(b,3,p,p)` and `gt` `(b,3,p·s,p·s)`. A missing required key raises at construction. Pool capacity is rounded to the batch. The first 3 batches equal the current `otf.feed_data` under the same seed (after the T3 root-key fix). |
| 1c | `neosr/models/tiling.py` (new) | §5 | `t2probe/tile_probe.py` as a test: `allclose(atol=1e-12)` fp64 for halo ≥ RF, plus a mismatch for halo < RF. Odd/tiny shapes. Output channels ≠ input channels. |
| 1d | deletions | 4 legacy model files, `.helaman`, `check_resume` | `python -c "import neosr.models"` (after T3 removes the import-time argv parse); grep `model_type` in options/ returns only image/otf |
| 2 | `phase.py` (T1 file) | §7.2 | Round-trip on a 2-conv net with AdamW + MultiStepLR + fp16 scaler: state equal. A type or name change raises with the first differing name. |
| 3 | `base.py`, `image.py` | §3 order, §4 counters, §5, §7.3, §7.4; T1 owns the body of `optimize_parameters`. Agree the seam (`self.update`, `self.micro`, `net_*_fwd`) before both edit `image.py`. | (i) CPU: save → new model → resume. Nets, EMA, phases, `update`, `best` and RNG are identical, and one `optimize_parameters` on an identical hand-fed batch gives identical weights in both. (ii) `ema=-1` with `adan_sf`: the export equals the `optimizer.eval()` weights, and live weights are `torch.equal` before/after export and validation. (iii) v1 fixture `3236.state` resumes with `update=3236`. (iv) GPU smoke on `options/train_esrgan2.toml` with a 50-image subset, `accumulate=2`, `ema=0.999`: 20 updates, SIGINT, resume to 40; exactly one boundary state is written per stop. Repeat with `use_amp` fp16 and `compile=true`: no `_orig_mod.` in any saved key, and `test.py` loads the export. |
| 4 | `otf.py` | §7.5 | Same smoke on one otf config |
| 5 | `train.py`, `test.py`, `convert.py`, `neosr/utils/logger.py:143,151` | §7.6; delete `load_resume_state`, the throughput kill (`577-584`) and the duplicate close; `convert.py` uses `read_weights`; the logger stops dividing | `test.py` on `net_g_2000.pth` reproduces the previous PSNR (non-tiled). An explicit missing/corrupt `resume_state` raises and the experiment folder is not renamed. Resume with `update >= total_iter` exits without reading data. |
| 6 | docs | `val.tile_overlap`, `logger.keep_states`, compile/DDP policy, `.state` v2, resume policy and tolerance table | — |

## 9. Interfaces with other topics

- **T1:** `optimize_parameters() -> bool` increments `self.update` and passes `it=self.update` to the loss `Context`. It uses `self.net_g_fwd` for the G forward. `Ema` is built in §3 step 4. `Phase` gains §7.2. `check_finite()` runs before validate/save. `eval_weights()` (§5) is the single implementation T1 §2.6 refers to, and `.state` is not written inside it (T1 §2.6 "save snapshots state dicts inside it" is superseded: the state stays in train representation). Under DDP, T1 must call D in the G phase without DDP gradient sync, because D's params are frozen there and DDP does not tolerate `requires_grad` toggling on wrapped params. The proposal is the bare `self.net_d` in the G phase and `self.net_d_fwd` in the D phase. This needs one DDP smoke test before it is relied on (risk).
- **T3:** owns `utils/rng.py` (one in-place-reseeded Generator, the source of `rng_state()["neosr"]`), config merge of `[degradations]`, removed keys (`ignore_resume_networks`, DP use, `enable_qat` per T4), entry points, and the explicit device (no `set_default_device`).
- **T4:** OPTIMIZERS table, compile policy (`compile_mode`, `dynamic=False`), pool swap, removed features (SAM, D-EMA, QAT trainer path).

## 10. Risks

- Exact-continuation scope: weights, EMA, optimizers, schedulers, scalers, counters and rank-0 main-process RNG are exact. Data order inside the resumed epoch, worker augmentation streams, prefetched batches and the OTF pool are not, and the resume log line says so. Exact data replay needs a consumed-sample cursor and per-sample seeds (Codex §RNG), which is an owner decision.
- Each `.state` grows by 2 G + 1 D copies; `keep_states` mitigates.
- Legacy v1 resume continues live training from EMA weights when EMA was on (unavoidable: live weights were never saved), with a fresh fp16 scale and reseeded RNG. One warning is logged.
- EMA keeps averaging the schedule-free `y` sequence (status quo for 85 EMA configs). Averaging `x` would change results and is a one-line change in `Ema` if wanted.
- An optimizer that stores Python objects in its state fails `weights_only=True` at the first save test (soap_sf is deleted by T4; adan_sf/adamw_sf store tensors).
- Tiled inference is exact only for receptive field ≤ `tile_overlap`. Global-attention archs keep faint seams, as today, minus the CPU round trip.
- DDP: a rank-0 validation exception leaves the other ranks waiting at the barrier until the NCCL timeout. This is acceptable because the run is dead either way, and the last checkpoint stands.
- lr changes in the TOML do not apply on resume because optimizer groups are restored (status quo; documented).

## 11. Rejected (and from whom)

- Codex: an `ImageTrainer` + preparation-protocol + factory + 8-file layout (`runtime.py`, `qat.py`, `inference.py`, `validation.py`). It expresses the same two batch paths with more types (D1).
- Codex: a 7-field `Progress`, separate per-optimizer update counters, `latest.json`, a `provenance` block, per-rank gathered RNG, persisted OTF pool, DP retention, and partial-continuation `ignore_resume_networks`.
- Fable: `AveragedModel` (D6), an exterior/ragged tiling variant (D13), `meta` in the export, dropping `net_d_*.pth`, a QAT hook in the trainer, newest-first fallback in auto-resume, a separate `build` block, and persisting `micro`.
- Opus: `params_ema` alongside `params` in the export, and keeping the degradation code inside `otf.py`.
- All: mid-window saves, partial-gradient persistence, async saving, safetensors as default, a generic N-network abstraction, keeping `default.py` "for reference" (git has it).

## 12. Open questions for the owner

1. Exact data-order resume (sampler cursor + per-sample seeds + persisted pool), or is stochastic continuation enough?
2. `keep_states` default: keep all (status quo) or the newest 3?
3. QAT: confirm removal from the trainer (T4), or keep the §3 step-3 slot for aether?
4. Should `.state` also be accepted as a `pretrain_network_*` source (start a new run from another run's live weights)? It is 3 lines in `read_weights`; not included by default.
