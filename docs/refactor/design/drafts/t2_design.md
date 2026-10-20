# T2 design: neosr/models structure, validation, checkpoint/resume (final)

Repo: /home/korvin/apps/neosr-dev (paths relative). Every `path:line` is current code. [probe] = verified on the installed torch 2.9.1+cu128 with `.venv/bin/python -B`. Two independent Codex consults (design questions, then adversarial review of the draft) are folded in; §11 lists what was adopted and rejected.

## 0. Decisions (one line of rationale each)

| # | Decision | Rationale |
|---|---|---|
| D1 | Three model files survive: `base.py` (plumbing), `image.py` (SISR trainer), `otf.py` (feed_data override only). Delete `default.py`, `simple_otf.py`, `eqr_otf.py`, `pancam_otf.py`, `image.py.helaman`. | Only `image` (93 cfgs) and `otf` (42 cfgs) appear in options/; the legacy tree is a second optimizer/save/validation implementation with its own crashes (default.py:299,355,455) and `models/__init__.py:23` imports all of it at startup. |
| D2 | OTF degradation moves out of the model into `neosr/data/otf_pipeline.py` (`Degrader` + `PairPool`); `otf.feed_data` becomes ~15 lines. | Data synthesis is not training logic; one validated config object ends the root-vs-dataset key split (otf.py:146,154,207,231 read root keys, 116-138 read `datasets.train`). |
| D3 | Two artifacts: `models/net_g_<update>.pth` = inference export `{"params": sd, "meta": {...}}`; `training_states/<update>.state` = self-contained resume state v2 (live G, live D, EMA + `n_averaged`, optimizers, schedulers, scalers, counters, RNG, best metrics, construction metadata). `net_d_*.pth` is no longer written. | Resume must never read the export: today EMA weights are resumed as live weights paired with live optimizer moments (image.py:992-1003 → misc.py:151-159). D exists only to resume. |
| D4 | File names and every user-facing number count optimizer updates; the state stores `update` and `micro`. Saves happen only at update boundaries; a graceful stop is deferred to the next boundary. | Fixes save-as-updates/resume-as-microbatches (train.py:600 vs 460; logger.py:143 divides again). An interrupt between the G step and the D step (image.py:748-751) or inside FSAM's perturbation (fsam.py:80) is not a state that can be saved. |
| D5 | Keep `torch.optim.swa_utils.AveragedModel` for the G EMA, constructed from the **bare** module after the topology is final (QAT prepared and calibrated), touched only through `.module` (export/validation) and `.n_averaged` (state). Drop the discriminator EMA. | [probe] 2.9.1 `update_parameters` copies on the first update and syncs buffers every update with `use_buffers=False`, so no custom class; today the EMA deep-copies the compiled/DDP wrapper (image.py:98-102) and leaks `module.`/`n_averaged` into exports (base.py:346-349); the EMA-of-D overwrites the resumed D. Per-channel QAT observer buffers change shape after the first forward, so the copy must happen after calibration or the buffer sync raises. |
| D6 | Canonical module = bare (`DDP/DP .module`, then `_orig_mod`, to a fixed point). All state goes in/out through it. Resume happens inside `__init__`; wrapping (`compile`, `DDP`) is the last step. | Compile-before-load breaks keys (base.py:163 vs 229-235); optimizers built before QAT hold dead params (train.py:431 after image.py:115). [probe] `load_state_dict` into `_orig_mod` after compile is seen by the compiled graph. |
| D7 | Schedule-free protocol: `.state` is always written/read in **train** representation; export without EMA and validation without EMA materialize the `x` weights by snapshot → `optimizer.eval()` → use → `p.copy_(snapshot)` + `train_mode = True` in `finally`. EMA keeps tracking the train (`y`) weights. | `train_mode` is persisted in `param_groups` (adamw_sf.py:66,85,97) so representation must be pinned; today the swap happens after the copy (base.py:343 vs 353) and never in validation (image.py:851,889). `eval()`→`train()` is two in-place lerps, not guaranteed bit-inverse (Codex measured 6e-8 drift; my trial was exact; the snapshot makes it exact by construction [probe]). EMA-of-`x` would change results for 170+ configs. |
| D8 | Atomic writes: `torch.save` → unique `.tmp` in the same dir → fsync file → `os.replace` → fsync dir; cleanup on any exception, retry only `OSError`, 3 attempts, raise on final failure; `.state` first, `.pth` second; latest-state discovery ignores `.tmp`; resume failure raises. | Partial files, the retry that reports success as failure (base.py:361-377), and the silent fresh start that archives the folder (train.py:235-238, 353) are the three ways a run is lost today; a mid-write failure inside `torch.save` surfaces as `RuntimeError`, not `OSError` (Codex probe). |
| D9 | Pretrained loading: explicit `param_key` wins or errors; otherwise `params_ema` > `params-ema` > `params` > bare dict; strip `module.`/`_orig_mod.`; drop `n_averaged`; `strict=False` drops shape-mismatched keys with a warning; no `mmap`. One function shared by train/test/convert. | base.py:413-424 overrides the requested key and swallows errors; convert.py:95-104 has a different order; `mmap=True` rejects legacy non-zip files (torch/serialization.py:1537). |
| D10 | Validation runs on the bare eval net (fp32, `inference_mode`, inside `torch.random.fork_rng`, never through DDP, `dist.barrier()` around it); tiled path = tiles with context **clipped to the image** (no exterior padding), output crop by actual offsets, channels inferred from the first tile, CPU accumulator by default. | DDP forward on rank 0 only can hang on buffer broadcast (image.py:823); per-tile `empty_cache` and input shuttling (image.py:787-807) are the hot spots; reflect pad fails when pad ≥ dim (A28) and an exterior reflect halo changes the model's own boundary behaviour (Codex probe: 3×3 zero-padded conv corner 4 vs 9); DataLoader iteration consumes torch RNG, so validation must not perturb training RNG. |
| D11 | Legacy v1 `.state` + `net_g_<it>.pth`/`net_d_<it>.pth` still resume through one adapter (weights into live and EMA; warns that scaler/RNG/EMA count are reset). Every old `.pth` layout still loads as pretrained. | Owner requirement; experiments/train_esrgan holds five v1 states ([probe] `3236.state` loads under `weights_only=True`). |
| D12 | QAT is a duck-typed structural hook (`prepare_qat`, `calibrate`) in a fixed slot of `image.__init__`; the state stores the QAT config; resume and `test.py` re-run `prepare_qat` from the TOML before loading tensors and compare it with the stored config. | Only aether implements it (aether_arch.py:533-600); its fusion/quant flags are Python attributes (aether_arch.py:530,551) and `calibrate` leaves the net in eval mode (aether_arch.py:590), so topology is rebuilt from config, never from tensors. |

## 1. File layout

```
neosr/models/
  __init__.py      registry scan + build_model (API unchanged)
  base.py          class base   ~300 lines: device/wrap/bare, optimizer+scheduler factories, warmup,
                                checkpoint orchestration (save/resume/load_pretrained), validation driver, best metrics
  image.py         class image  ~450 lines: nets, EMA, QAT slot, feed_data, train_step, infer/infer_tiled, validate, export
  otf.py           class otf    ~40 lines: feed_data = degrade -> crop -> pool -> augment
  checkpoint.py    functions    ~180 lines: strip_prefixes, canonical_state_dict, read_weights, load_into, atomic_save,
                                find_latest_state, rng_state/set_rng_state, STATE_FORMAT = 2, upgrade_legacy_state, check_compatible
neosr/data/
  otf_pipeline.py  Degrader (two-stage RealESRGAN degradation, config validated once, owns DiffJPEG), PairPool (pair queue)
```

Deleted: `neosr/models/default.py`, `simple_otf.py`, `eqr_otf.py`, `pancam_otf.py`, `image.py.helaman`; `neosr/utils/misc.py:check_resume` (132-167) and its export (`neosr/utils/__init__.py:25,50`); in train.py the QAT block (414-446), throughput kill (577-584), duplicate close (656-659), `load_resume_state` (192-239). Datasets `simple_otf_dataset.py`, `pancamotf_dataset.py`, `pancamsingle_dataset.py` lose their only consumers (data topic decides; pancam is broken at pancam_otf.py:39,56 and has no config); `eqrotf_dataset.py` returns the same keys as `otf` (eqrotf_dataset.py:219 vs otf_dataset.py:261-267) and keeps working with `model_type = "otf"`.

## 2. Class structure and public methods

```python
class base:                                   # neosr/models/base.py
    def __init__(self, opt)                    # device, is_train, accumulate, counters; nets/optimizers/schedulers/scalers = {}
    def to_device(self, net) -> nn.Module      # .to(device, channels_last if use_amp)
    def wrap(self, net) -> nn.Module           # dist: compile(DDP(net)); else compile(net); returns the forward-callable
    @staticmethod
    def bare(net) -> nn.Module                 # unwrap .module / ._orig_mod until fixed point
    def build_optimizer(self, cfg, params) -> Optimizer   # SAM wrapper is THE registered optimizer when enabled; records initial_lr
    def build_schedulers(self) -> None
    def step_schedulers(self) -> None          # once per update; linear warmup from initial_lr
    def current_lr(self) -> list[float]
    def load_pretrained(self, net, path, param_key, strict) -> None
    def training_state(self, epoch) -> dict
    def save(self, epoch, final=False) -> None
    def resume(self, path) -> int              # returns epoch; raises on any failure
    def validation(self, loader, current_iter, tb_logger, save_img) -> None   # keyword name kept for test.py:55
    def get_current_log(self) -> dict[str, float]

class image(base):                            # neosr/models/image.py
    def __init__(self, opt)                    # §3 order, resume included
    def feed_data(self, data: dict) -> None    # training batch -> self.batch = {"lq", "gt"} (augmented)
    def train_step(self) -> bool               # one micro-batch; True when an optimizer update happened
    update_step: int; micro_step: int; start_epoch: int
    def eval_net(self) -> nn.Module            # ema.module if ema else bare(net_g)
    @contextmanager eval_weights(self)         # SF snapshot/swap when no EMA; eval()/restore mode; QAT observers frozen
    def infer(self, lq) -> Tensor              # direct or tiled
    def infer_tiled(self, net, lq, tile, halo, device) -> Tensor
    def validate(self, loader, update, tb_logger, save_img) -> None
    def export_state_dict(self) -> OrderedDict # bare, cpu, contiguous: EMA | SF-x | live
    def _state_extra(self) -> dict / _load_extra(self, state)   # ema, scalers

class otf(image):                             # neosr/models/otf.py
    def __init__(self, opt)                    # Degrader(cfg, scale, device), PairPool(queue_size, device)
    def feed_data(self, data) -> None          # "lq" in data: paired path; else degrade -> crop -> pool -> augment
```

Removed from the classes: the `is_train` flag flip for validation (image.py:851,980; otf.py:290-292) — validation never calls `feed_data`; `get_current_gan_weight`/`update_loss_weights` hooks (base.py:121-145) — the T1 loss pipeline logs the GAN weight in its own log dict; `reduce_loss_dict` per-iteration `.item()` (base.py:559) — the model keeps detached device-side sums and `get_current_log()` converts at print time (contract owned by T1/perf).

## 3. Initialization order (`image.__init__`)

```
1. net_g = to_device(build_network(opt["network_g"]))                       # eager, bare
2. if path.pretrain_network_g and not resuming: load_pretrained(net_g, path, path.param_key_g, path.strict_load_g)
3. if train.enable_qat: net_g.prepare_qat(qat_cfg); net_g.calibrate(loader) if calibration_iters and not resuming
                        else one no_grad forward on a dummy patch; net_g.train()        # topology AND buffer shapes final
4. ema = AveragedModel(net_g, multi_avg_fn=get_ema_multi_avg_fn(train.ema), device=device) if train.ema > 0 else None
5. same 1-2 for net_d (no QAT/EMA); optimizers from bare parameters(); scalers; schedulers; losses (T1)
6. if resuming: self.resume(opt["path"]["resume_state"])                   # validates, then loads into bare modules
7. self.net_g = wrap(net_g); self.net_d = wrap(net_d)                       # compile / DDP last
```
Invariant: everything that captures parameter identity or buffer shape (EMA copy, optimizer) is created after the last structure change and before wrapping. `resume()` loads into `bare()` modules; compiled graphs share the tensors [probe]. In DDP, resuming before wrap means DDP's constructor broadcast starts from identical weights on every rank.

Pretrained load is skipped when resuming (the state carries the weights); this replaces `check_resume` rewriting `pretrain_network_*` to the `.pth` files (misc.py:151-159). `test.py` (`is_train=False`) runs steps 1-3 and 7: a QAT export carries the prepared topology's keys, so loading it needs the same `enable_qat` config (documented; `convert_to_quantized` is a separate export path, out of scope).

## 4. Counters, update boundary, graceful stop

| Counter | Meaning | Persisted |
|---|---|---|
| `micro_step` | micro-batches consumed (`train_step` calls) | `state["micro"]` |
| `update_step` | completed logical updates (every `accumulate` micro-batches; an fp16 scaler-skipped step still counts, so schedules keep their length) | `state["update"]`, file names |
| `ema.n_averaged` | EMA updates applied; skipped when the fp16 scaler skipped (`scaler.get_scale()` dropped after `update()`; one float read per update) | `state["ema"]["n_averaged"]` |
| `epoch` | sampler epoch | `state["epoch"]` |

`train_step()` returns `stepped = micro_step % accumulate == 0`. train.py keys logging, validation, saving and `step_schedulers()` on `stepped` and uses `model.update_step` for everything user-facing; ECO/SAM/wavelet thresholds compare against `update_step` (today they receive the micro count, train.py:552 → image.py:511,538,626, and fire `accumulate`× early).

Graceful stop: train.py installs a SIGINT handler that sets `stop_requested`; the loop checks it right after a completed update (after scheduler step and EMA), validates if due, saves, exits 0. A second SIGINT restores the default handler (hard exit; the previous committed checkpoint stands). No mid-window or mid-transaction save exists.

Epoch budgeting in train.py:149-155 mixes units (`num_iter_per_epoch` in updates, `total_iters` in micro-batches); the new loop computes `updates_per_epoch = len(sampler) // (batch * accumulate)` and `total_epochs = ceil(total_updates / updates_per_epoch)`; a resume with `update_step >= total_updates` exits before touching data.

## 5. Validation

```python
def eval_net(self):
    return self.ema.module if self.ema is not None else self.bare(self.net_g)

@contextmanager
def eval_weights(self):
    net = self.eval_net()
    swap = self.ema is None and self.is_train and self.sf_optim_g
    was_training = net.training
    snapshot = [p.detach().clone() for p in net.parameters()] if swap else None
    if swap: self.optimizer_g.eval()                   # p.data <- x (adamw_sf.py:75-85)
    net.eval()
    if self.qat: torch.ao.quantization.disable_observer(net)
    try:
        yield net
    finally:
        if self.qat: torch.ao.quantization.enable_observer(net)
        net.train(was_training)
        if swap:                                       # exact restore, independent of lerp round-off [probe]
            with torch.no_grad():
                for p, s in zip(net.parameters(), snapshot): p.copy_(s)
            for g in self.optimizer_g.param_groups: g["train_mode"] = True

def infer(self, lq):
    val = self.opt.get("val") or {}
    with self.eval_weights() as net, torch.inference_mode(), torch.autocast("cuda", enabled=False):
        lq = lq.to(self.device, memory_format=self.memory_format, non_blocking=True)
        tile = val.get("tile", -1)
        return net(lq) if tile == -1 else self.infer_tiled(net, lq, tile, val.get("tile_overlap", 16), val.get("tile_stitch", "cpu"))
```

Tiled inference (context clipped to the image, so exterior boundaries see exactly what direct inference sees):
```python
def infer_tiled(self, net, lq, tile, halo, device):
    b, c, h, w = lq.shape; s = self.scale; out = None
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            iy0, ix0 = max(y0 - halo, 0), max(x0 - halo, 0)                       # context, never beyond the image
            iy1, ix1 = min(y1 + halo, h), min(x1 + halo, w)
            y = net(lq[:, :, iy0:iy1, ix0:ix1])
            y = y[:, :, (y0 - iy0) * s : (y1 - iy0) * s, (x0 - ix0) * s : (x1 - ix0) * s]
            if out is None:                                                        # channels/dtype from the first tile
                out = torch.empty((b, y.shape[1], h * s, w * s), dtype=y.dtype, device=device)
            out[:, :, y0 * s : y1 * s, x0 * s : x1 * s] = y.to(device, non_blocking=True)
    return out
```
Ragged last tiles are passed as-is; archs that need window multiples already pad internally (swin-style `check_image_size`), and direct inference has the same property. `tile_stitch = "cpu"` keeps today's memory profile (image.py:792) without the per-tile input shuttle and `empty_cache`; `"cuda"` is opt-in for speed when the output fits.

`validate()`: one loop; `sr = self.infer(data["lq"])`; metric modules cached (metrics topic); metric values stored as `float()` (PSNR returns a numpy scalar, calculate.py:65, which `weights_only=True` refuses); `best_metric_results` restored from the state so "Best @ iter" survives resume; empty loader rejected at startup. `validation()` in `base`: `with torch.random.fork_rng(devices=[self.device]):` so the validation loader's RNG use never touches training RNG, `dist.barrier()` before and after, rank 0 only computes, always through the bare/EMA module. Order at a boundary: validate (if due) → save (if due), so the saved `best` already includes that update.

## 6. Checkpoint format

### 6.1 `models/net_g_<update>.pth` (user-facing)
```python
{"params": OrderedDict[str, Tensor],     # bare keys, cpu, contiguous: EMA if ema>0 else SF-x / live
 "meta": {"neosr": 2, "arch": "esrgan", "scale": 4, "update": 3000, "ema": True}}   # JSON-only values, additive
```
Key stays `"params"` (what every existing experiment and chaiNNer/spandrel-style loaders expect; convert.py:95-104 and base.py:414-419 both unwrap it). `meta` is additive; any loader indexing `["params"]` is unaffected. `net_g_latest.pth` is written at the end of training; `save(epoch, final=True)` also writes the final `.state` (train.py:628 + base.py:461 suppress it today).

### 6.2 `training_states/<update>.state` (format 2)
```python
{
 "format": 2, "epoch": int, "update": int, "micro": int, "name": opt["name"],
 "build": {"accumulate": int, "use_amp": bool, "bfloat16": bool, "ema": float, "qat": dict | None,
           "network_g": opt["network_g"], "network_d": opt["network_d"] | None,
           "optimizer_types": {"net_g": "adan_sf", "net_d": "adan_sf"}, "world_size": int},
 "nets":       {"net_g": sd, "net_d": sd},                   # LIVE bare weights; SF: train representation
 "ema":        {"net_g": {"params": sd, "n_averaged": int}}, # absent when ema <= 0
 "optimizers": {"net_g": sd, "net_d": sd},                   # name-keyed; SF: every group train_mode == True; SAM wrapper includes its base optimizer and momentum
 "schedulers": {"net_g": sd, "net_d": sd},
 "scalers":    {"net_g": sd, "net_d": sd},                   # GradScaler.state_dict(); {} when disabled [probe]
 "rng":        {"python": random.getstate(), "numpy": {"key": Tensor(uint32), "pos": int, "has_gauss": int, "gauss": float},
                "neosr": generator.bit_generator.state, "torch": ByteTensor, "cuda": [ByteTensor]},   # rank 0 only
 "best":       {dataset: {metric: {"better": str, "val": float, "iter": int}}},
}
```
Rules: must load with `torch.load(weights_only=True)` — [probe] python/torch/cuda RNG, optimizer dicts and nested dicts load; `np.random.get_state()` only if its ndarray is stored as a tensor; the neosr `default_rng` is PCG64 whose `bit_generator.state` is plain ints (rng.py:14); numpy scalars are refused (hence `float()` in §5). A unit test round-trips a state produced by a real validation. Self-contained: resume needs exactly one file. Size: old layout = export G(EMA) + `net_d.pth`(D EMA) + `.state`(2 optimizers); new = export G(EMA) + `.state`(G live, G EMA, D, 2 optimizers) → two extra G copies, one D copy fewer. `logger.keep_states = N` prunes old `.state` files after a successful write (default: keep all, as today).

Contract for the optimizer topic: exactly one optimizer object per net is registered under `optimizers["net_g"]`; with SAM that object is the wrapper and its `state_dict()`/`load_state_dict()` must carry the base optimizer state and the FSAM momentum (`fsam.py:30-31,59`; today image.py:405 keeps `sam_optimizer_g` outside `self.optimizers`). SOAP must key its state by stable identifiers: a Codex probe shows its `(param, idx)` keys deserialize but stay bound to the old parameter objects (soap_sf.py:629), so "loads without error" is not the acceptance test — "next update equals the uninterrupted run" is.

Resume validation (`check_compatible(state, self)`, before anything is mutated): component sets equal (`nets`, `optimizers`, `schedulers`, `scalers`), `optimizer_types` equal, per-optimizer group count and per-group parameter count equal, `network_g/network_d` dicts equal, `qat` config equal, `use_amp`/`bfloat16` equal (a `{}` scaler state into an enabled scaler raises, Codex probe), `ema > 0` equal. `accumulate` may change: warn and set `micro = update * accumulate_new`. `world_size` may change: warn; ranks ≠ 0 reseed from `manual_seed + rank` as at startup (options.py:213), i.e. RNG is exact for rank 0 only.

### 6.3 Legacy (format 1) resume
`upgrade_legacy_state(state, opt, log)` when `"format"` is absent: `update = state["iter"]`, `micro = update * accumulate`, `nets` from `models/net_g_<iter>.pth` and `net_d_<iter>.pth` (what `check_resume` does today), EMA = the same G weights with `n_averaged = update`, scalers default, RNG untouched, optimizer/scheduler lists zipped with `("net_g", "net_d")`, `build` synthesized from the current config (so compatibility checks pass by construction and are logged as unverified). One warning line lists what was reset. Fixture: `experiments/train_esrgan/training_states/3236.state` + `models/net_{g,d}_3236.pth`.

### 6.4 Pretrained files accepted by `read_weights`
`{"params_ema"|"params-ema"|"params": sd}` (precedence in that order unless `param_key` is given), bare `sd`, any with `module.`/`_orig_mod.` prefixes, with or without `n_averaged`. `.safetensors` is an optional 4-line branch if the package is importable.

## 7. Code skeletons

### 7.1 `neosr/models/checkpoint.py`
```python
STATE_FORMAT = 2
_PREFIXES = ("module.", "_orig_mod.")
_EMA_WRAPPER_KEYS = {"n_averaged"}

def strip_prefixes(sd):
    out = {}
    for k, v in sd.items():
        while k.startswith(_PREFIXES):
            k = k.split(".", 1)[1]
        if k in _EMA_WRAPPER_KEYS: continue
        if k in out: raise KeyError(f"prefix stripping collides on {k!r}")
        out[k] = v
    return out

def canonical_state_dict(net):
    return OrderedDict((k, v.detach().to("cpu", memory_format=torch.contiguous_format))
                       for k, v in strip_prefixes(net.state_dict()).items())

def read_weights(path, param_key=None):
    obj = torch.load(path, map_location="cpu", weights_only=True)          # no mmap: legacy non-zip files
    if param_key is not None:
        if param_key not in obj:
            raise KeyError(f"{path}: param_key {param_key!r} not found; top-level keys: {list(obj)}")
        return strip_prefixes(obj[param_key]), param_key
    for key in ("params_ema", "params-ema", "params"):
        if key in obj: return strip_prefixes(obj[key]), key
    if all(isinstance(v, Tensor) for v in obj.values()): return strip_prefixes(obj), None
    raise KeyError(f"{path}: no recognised parameter key; top-level keys: {list(obj)}")

def load_into(net, sd, strict, log):
    if not strict:
        own = net.state_dict()
        for k in [k for k in sd if k in own and own[k].shape != sd[k].shape]:
            log.warning(f"shape mismatch, skipping {k}: {tuple(own[k].shape)} vs {tuple(sd[k].shape)}"); del sd[k]
    res = net.load_state_dict(sd, strict=strict)        # strict mismatch raises torch's own message
    for k in res.missing_keys: log.warning(f"missing key: {k}")
    for k in res.unexpected_keys: log.warning(f"unexpected key: {k}")

def atomic_save(obj, path, attempts=3):
    for attempt in range(1, attempts + 1):
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{attempt}.tmp")
        try:
            with open(tmp, "wb") as f:
                torch.save(obj, f); f.flush(); os.fsync(f.fileno())
            os.replace(tmp, path)
            dfd = os.open(path.parent, os.O_RDONLY)
            try: os.fsync(dfd)
            finally: os.close(dfd)
            return
        except OSError as e:
            tmp.unlink(missing_ok=True)
            if attempt == attempts: raise OSError(f"cannot write {path} after {attempts} attempts") from e
            time.sleep(1)
        except Exception:                                   # serialization errors are not retried
            tmp.unlink(missing_ok=True); raise

def find_latest_state(d):                                  # integer-named, finalized files only
    return max((p for p in d.glob("*.state") if p.stem.isdigit()), key=lambda p: int(p.stem), default=None)

def rng_state():
    s = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"name": s[0], "key": torch.from_numpy(s[1].copy()), "pos": s[2], "has_gauss": s[3], "gauss": s[4]},
            "neosr": neosr_rng.bit_generator.state, "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}

def set_rng_state(s): ...                                   # inverse; tolerant of missing keys (legacy)
def upgrade_legacy_state(state, opt, log): ...              # §6.3
def check_compatible(state, model): ...                     # §6.2, raises RuntimeError with the first mismatch
```

### 7.2 EMA usage (no new file)
```python
self.ema = AveragedModel(net_g, multi_avg_fn=get_ema_multi_avg_fn(decay), device=self.device)  # BARE module, after QAT+calibration
# per update:  if not scaler_skipped: self.ema.update_parameters(self.bare(self.net_g))
# export:      canonical_state_dict(self.ema.module)              # bare keys, no n_averaged
# state:       {"params": self.ema.module.state_dict(), "n_averaged": int(self.ema.n_averaged)}
# load:        self.ema.module.load_state_dict(s["params"]); self.ema.n_averaged.fill_(s["n_averaged"])
```
Never call `self.ema.state_dict()`/`self.ema(x)` elsewhere; the wrapper's prefixed keys and buffer-sync semantics stay inside these five lines.

### 7.3 `base.py` save / resume
```python
def save(self, epoch, final=False):
    assert self.micro_step % self.accumulate == 0, "save() only at update boundaries"
    upd = self.update_step
    state = self.training_state(epoch)                       # cheap on every rank; only rank 0 writes
    if self.opt["rank"] != 0: return
    atomic_save(state, self.paths.training_states / f"{upd}.state")          # authoritative first
    export = {"params": self.export_state_dict(), "meta": self._meta(upd)}
    atomic_save(export, self.paths.models / f"net_g_{upd}.pth")
    if final: atomic_save(export, self.paths.models / "net_g_latest.pth")
    self._prune_states()

def training_state(self, epoch):
    for o in self.optimizers.values():
        if "train_mode" in o.param_groups[0]:               # schedule-free: pin the representation
            assert all(g["train_mode"] for g in o.param_groups), "save() requires SF optimizers in train mode"
    return {"format": STATE_FORMAT, "epoch": epoch, "update": self.update_step, "micro": self.micro_step,
            "name": self.opt["name"], "build": self._build_info(),
            "nets": {n: canonical_state_dict(self.bare(net)) for n, net in self.nets.items()},
            "optimizers": {n: o.state_dict() for n, o in self.optimizers.items()},
            "schedulers": {n: s.state_dict() for n, s in self.schedulers.items()},
            "rng": rng_state(), "best": getattr(self, "best_metric_results", {}),
            **self._state_extra()}                           # image: ema, scalers

def resume(self, path):
    log = get_root_logger()
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != STATE_FORMAT:
        state = upgrade_legacy_state(state, self.opt, log)   # raises if the companion .pth files are missing
    check_compatible(state, self)                            # §6.2; nothing mutated before this passes
    for n, sd in state["nets"].items():       load_into(self.bare(self.nets[n]), sd, strict=True, log=log)
    for n, sd in state["optimizers"].items(): self.optimizers[n].load_state_dict(sd)
    for n, sd in state["schedulers"].items(): self.schedulers[n].load_state_dict(sd)
    self._load_extra(state)
    self.update_step = state["update"]
    self.micro_step = self.update_step * self.accumulate     # exact when accumulate is unchanged, consistent otherwise
    self.best_metric_results = state.get("best", {})
    if "rng" in state and self.opt["rank"] == 0: set_rng_state(state["rng"])   # last: nothing after this consumes randomness
    log.info(f"resumed {path.name}: update {self.update_step}, epoch {state['epoch']} "
             "(weights/optimizers/EMA/counters exact; data order, worker augmentation and OTF pool restart)")
    return state["epoch"]
```
`self.nets`/`self.optimizers`/`self.schedulers`/`self.scalers` are small name-keyed dicts filled by `image.__init__`; `self.net_g`/`self.net_d` remain for the training step. Optimizer loading matches parameters by position, so `build_optimizer` is deterministic (`named_parameters` order) and `check_compatible` compares group/parameter counts first.

### 7.4 `image.py` export and extras
```python
def export_state_dict(self):
    if self.ema is not None: return canonical_state_dict(self.ema.module)
    with self.eval_weights() as net: return canonical_state_dict(net)     # SF: x weights; plain: live

def _state_extra(self):
    extra = {"scalers": {n: s.state_dict() for n, s in self.scalers.items()}}
    if self.ema is not None: extra["ema"] = {"net_g": {"params": self.ema.module.state_dict(), "n_averaged": int(self.ema.n_averaged)}}
    return extra

def _load_extra(self, state):
    for n, sd in state.get("scalers", {}).items(): self.scalers[n].load_state_dict(sd)
    if self.ema is not None:
        e = state.get("ema", {}).get("net_g")
        if e: self.ema.module.load_state_dict(e["params"]); self.ema.n_averaged.fill_(e["n_averaged"])
        else: self.ema.module.load_state_dict(self.bare(self.net_g).state_dict()); self.ema.n_averaged.fill_(self.update_step)  # legacy
```

### 7.5 `otf.py`
```python
@MODEL_REGISTRY.register()
class otf(image):
    def __init__(self, opt):
        super().__init__(opt)
        cfg = opt["datasets"]["train"]                               # merged with opt["degradations"] by train.py:131-132
        self.degrader = Degrader(cfg, scale=self.scale, device=self.device)   # validates every key once; owns DiffJPEG
        self.pool = PairPool(cfg.get("queue_size", 180), device=self.device)  # capacity rounded to the runtime batch at first call

    @torch.no_grad()
    def feed_data(self, data):
        if "lq" in data: return super().feed_data(data)
        gt = data["gt"].to(self.device, non_blocking=True)
        lq = self.degrader(gt, data["kernel1"], data["kernel2"], data["sinc_kernel"])
        gt, lq = paired_random_crop([gt], lq, self.patch_size, self.scale)   # patch_size is the LR size (transforms.py:117-118)
        lq, gt = self.pool(lq, gt)
        self.batch = self._augment({"lq": lq.contiguous(), "gt": gt})
```
`Degrader.__call__` is otf.py:105-252 with every key read from `cfg` and `jpeg_compress` honoured in both orders (otf.py:235-240 ignores it). The pool is empty after resume (policy, §9).

### 7.6 `train.py` loop (existing `CUDAPrefetcher.reset()/next()` API kept, prefetch_dataloader.py:104)
```python
opt["path"]["resume_state"] = resolve_resume(opt)     # explicit path (must exist) or auto_resume newest loadable .state, else None
if opt["path"]["resume_state"] is None: make_exp_dirs(opt)   # never archive a folder we are resuming
model = build_model(opt)                               # resume happened inside __init__
stop = install_sigint_flag()                           # first Ctrl-C: stop at the next boundary; second: default handler
done = model.update_step >= total_updates
for epoch in range(model.start_epoch, total_epochs + 1):
    if done: break
    sampler.set_epoch(epoch); prefetcher.reset(); batch = prefetcher.next()
    while batch is not None and not done:
        model.feed_data(batch)
        if model.train_step():
            upd = model.update_step
            model.step_schedulers()
            if upd % print_freq == 0: msg_logger(epoch, upd, model.current_lr(), model.get_current_log())
            due_val, due_save = val and upd % val_freq == 0, upd % save_freq == 0
            done = upd >= total_updates or stop.is_set()
            if due_val or done: for loader in val_loaders: model.validation(loader, upd, tb_logger, save_img)
            if due_save or done: model.save(epoch, final=done and not stop.is_set())
        batch = prefetcher.next()
```
Other exceptions propagate with their traceback (no `sys.exit(1)` wrapper, train.py:637-642). `resolve_resume` validates candidates newest-first (`torch.load(weights_only=True)` + `format` check), reports a fallback explicitly, and fails on an explicitly requested corrupt file. `MessageLogger` stops dividing by `accumulate` (logger.py:143,151) because it now receives updates.

## 8. Migration steps (ordered; letters in one row are parallel-safe, disjoint files)

| Step | Files | Work | Verification |
|---|---|---|---|
| 1a | `neosr/models/checkpoint.py` (new) | §7.1 | pytest (CPU): prefix stripping incl. `_orig_mod.module.`, collision raises, key precedence, explicit bad key raises; `atomic_save` leaves no `.tmp` and raises after 3 injected `OSError`s, cleans up on an injected `RuntimeError` without retry; `find_latest_state` ignores `.tmp`; `read_weights` on `experiments/train_esrgan/models/net_g_2000.pth`; `rng_state()` and a `best` dict from real validation round-trip under `weights_only=True`. |
| 1b | `neosr/data/otf_pipeline.py` (new) | `Degrader`, `PairPool` lifted from otf.py:37-252 | pytest with a seeded config: `lq` shape `(b,3,patch,patch)`, `gt` `(b,3,patch*s,patch*s)`; both JPEG orders honour `jpeg_compress=false`; a cfg missing `poisson_scale_range` raises at construction; pool capacity rounds to the runtime batch; first 3 batches equal old `feed_data` under the same seed (except the fixed JPEG-order bug). |
| 1c | deletions | `default.py`, `simple_otf.py`, `eqr_otf.py`, `pancam_otf.py`, `image.py.helaman` | `python -c "import neosr.models"` (after the data topic removes the import-time CLI parse; until then run with `-opt`); grep options/ for `model_type` still only image/otf. |
| 2 | `base.py` | §7.3; `bare()`; `to_device`/`wrap`; name-keyed dicts; `step_schedulers` with `initial_lr` recorded at build; `check_compatible` | pytest: `bare()` on DP→compile→DDP mock; save→resume round-trip on a 2-conv net with AdamW+CosineAnnealing: optimizer/scheduler/counters/best/RNG identical; precision or optimizer-type change raises before mutation; v1 fixture (`train_esrgan/3236.state` + pths) resumes with `update=3236`, EMA `n_averaged=3236`. |
| 3 | `image.py` | §3 order, §4 counters, `train_step` → `stepped`, §5 validation, §7.4 export/extras; T1 owns the body of `train_step`, T2 everything around it (agree the seam before editing the same file) | GPU smoke on `options/train_esrgan2.toml` with a 50-image subset, deterministic, `accumulate=2`, `ema=0.999`, `adan_sf`, `num_worker_per_gpu=0`: (i) 40 updates straight; (ii) 20, SIGINT, resume to 40 → `net_g_40.pth` tensors and optimizer states `torch.equal` **when the data topic's single shared generator is in place**; until then the acceptance test is: resumed weights/optimizer/EMA equal the saved ones, and one `train_step` on an identical hand-fed batch produces identical gradients in both runs. Repeat with fp16 `use_amp` (scaler state) and `ema=-1` (export equals `optimizer.eval()` weights; live weights `torch.equal` before/after export and after a validation). Tiled vs direct inference on one image: `torch.equal` for a zero-padded CNN with `halo=16`. |
| 4 | `otf.py` | §7.5 | Same smoke on one otf config; resume equivalence holds except the first `queue_size/batch` batches (pool refill, documented). |
| 5 | `train.py`, `test.py:55`, `convert.py`, `neosr/utils/logger.py:143,151`, `neosr/utils/__init__.py:25,50` | §7.6; remove QAT block, kill switch, duplicate close, `load_resume_state`; delete `check_resume`; `convert.load_net` uses `read_weights`; logger stops dividing | `test.py` on `net_g_2000.pth` reproduces the previous PSNR; `convert.py` produces the same ONNX graph hash for one arch; an aether QAT config reaches optimizer construction after `prepare_qat` (import-level check; aether registration is G8.4); SIGINT during a window saves exactly one boundary state; resume with `update >= total_iter` exits without reading data. |
| 6 | docs | `val.tile_overlap`, `val.tile_stitch`, `logger.keep_states`, `.state` v2 layout, resume policy, QAT export contract | — |

Interfaces with other topics: T1 owns `train_step` internals and the log-dict contract; the data topic owns `neosr.utils.rng` as one injected generator with a state getter/setter (today `rng()` returns a fresh independent `Generator` per module and parses the CLI at import, rng.py:11-14, so `rng["neosr"]` cannot be complete until then) and, if exact order is wanted, `EnlargedSampler.set_epoch(epoch, start_index=0)`; the optimizer topic owns SAM wrapper serialization and SOAP's stable state keys.

## 9. Risks

- EMA keeps tracking the SF `y` weights (status quo); documented, not changed.
- Two extra G copies per checkpoint inside `.state`; `keep_states` mitigates.
- Legacy resume resets the fp16 scaler, EMA count and RNG; the first updates after a legacy resume are not bit-reproducible. One explicit warning.
- `weights_only=True` on `.state` refuses optimizers that store Python objects (SOAP today) — surfaces at the first save test, by design.
- Compiled training net vs eager EMA/validation net: validation is slower than a compiled path, acceptable at `val_freq`; `EMA.module` can be wrapped lazily if needed.
- `meta` in the export: a loader that takes `next(iter(d.values()))` blindly could pick it; spandrel/chaiNNer resolve keys by name. Dropping `meta` is one line.
- Clipped-context tiling is exact for receptive field ≤ `halo` LR px; global-attention archs keep faint seams (as today, minus the CPU shuttle); ragged last tiles rely on the arch's own size handling.
- Resume policy is "stochastic continuation": weights/optimizers/EMA/counters/rank-0 main-process RNG exact; worker augmentation streams, prefetched batches and the OTF pool are not. Stated in the resume log line.
- `fork_rng` around validation costs one CUDA RNG state copy per validation; negligible.

## 10. What I would NOT do

- A single-file checkpoint: the export must stay a small, bare, tool-compatible `.pth`.
- Separate `resume_models/net_*.pth` files (competitor layout, sr_model.py:1200-1225): more files to keep consistent for no gain over one self-contained `.state`.
- A custom EMA class: `AveragedModel` already copies on first update and syncs buffers [probe]; the wrapper only needs to stay behind `.module`/`.n_averaged`.
- Mid-window or mid-transaction saves, partial-gradient persistence, persisting the OTF `PairPool`, DataLoader worker RNGs or prefetcher position.
- Discriminator EMA, EMA-of-`x` for schedule-free, safetensors as default, new mandatory config keys, renamed keys.
- A generic N-network abstraction: two nets, two attributes, small name-keyed dicts.
- Keep `default.py` "for reference": git has it.

## 11. Codex second opinion (gpt-6.1-sol, read-only, two fresh threads)

Design consult — agreed independently: self-contained `.state` + bare `.pth` export (over the competitor's split), no D EMA, update-count naming with boundary-only saves, train-representation SF snapshots with `x` materialized for export/validation, tmp+fsync+replace, halo-and-crop tiling, rank-0 validation on the bare module with barriers. Disagreed on my draft's custom EMA → I checked `AveragedModel` on 2.9.1 and switched (D5). Adopted from it: directory fsync and unique tmp per attempt; `.state` before `.pth`; final completion writes a state; QAT Python flags force topology rebuild from config; epoch-unit mix in train.py:149-155; SAM's private base optimizer never saved; rank-local RNG gathered outside the rank-0 write; QAT observers frozen during validation.

Adversarial review of the draft (16 findings) — adopted: exterior reflect halo changes boundary behaviour → clipped context (§5); numpy scalars in `best` break `weights_only` → `float()`; SF `eval()/train()` not guaranteed bit-inverse → snapshot/restore (D7, [probe] exact); validation loader consumes torch RNG → `fork_rng` and validate-then-save order; `KeyboardInterrupt` mid-transaction → deferred stop flag (§4); `mmap=True` rejects legacy files → dropped; serialization errors bypass `OSError` retry → cleanup on any exception; resume must validate component sets/types/precision before mutating → `check_compatible`; prefetcher `next/reset`, `test.py:55` kwarg, `logger.py:143`, `utils/__init__.py:25` callers listed in step 5; OTF test shape used the HR contract → LR (transforms.py:117-118); storage accounting corrected to +2 G −1 D; QAT observer buffers resize after the first forward → EMA created after calibration; exact-resume criterion gated on the data topic's single generator, with a hand-fed-batch gradient test until then; CPU tile accumulator as default. Rejected: a separate "successful optimizer calls" counter (optimizer `step` state and `n_averaged` already carry it); rejecting SAM resume outright (the wrapper-serialization contract is cheaper).

## 12. Open questions for the owner

1. Does any downstream tool read `net_d_*.pth`? If yes: opt-in `logger.export_discriminator = true`.
2. `keep_states` default: keep all (status quo) or 3?
3. Exact data-order resume (sampler cursor + single generator in the data topic) — wanted, or is stochastic continuation enough?
4. QAT: keep the duck-typed hook (aether only, and aether_* registration is currently broken, G8.4) or drop QAT support entirely in this refactor?
