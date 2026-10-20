# neosr-dev deep review (read-only)

Scope covered: train.py, test.py, convert.py, neosr/models, losses, data, utils, metrics, optimizers, archs helpers + discriminators + vgg_arch, logs in experiments/, competitor traiNNer-redux-dev core. Claims marked **[verified]** were checked against the installed torch 2.9.1 in `.venv` with small CPU/GPU experiments; everything else is from reading the code.

Log evidence used throughout (`experiments/train_esrgan/train_train_esrgan_20251230_115003.log`): by iter 1000 `out_d_real` = 12 → 39..47, `out_d_fake` = −24 → −37..−45, `l_d_total` = 1e-3 → 1e-14, `l_g_fake`/`l_d_fake` hit exactly 0.0, while `l_g_gan` grows to 9–13 and becomes ~60 % of `l_g_total` with no visible effect on the other G losses. That is a discriminator that has won completely against a generator that cannot fight back; sections A1/A2 explain why.

## 1. Top 10 issues

1. **Generator never receives adversarial gradient.** `image.py:610-612` runs `net_d(self.output)` under `torch.inference_mode()`; `l_g_gan.requires_grad` is `False` **[verified]**, so `l_g_total += l_g_gan` adds a constant. GAN training is dead for G; D saturates (see logs). Critical.
2. **Discriminator is trained in `eval()` mode.** `image.py:654` sets `net_d.eval()` before the D forward/backward. For `metagan`/`unet`/`dunet`/`resgan`/`ea2fpn` (all `spectral_norm`) the power iteration is skipped in eval **[verified: `weight_u` unchanged in eval, changes in train]**, so sigma stays at its random-init estimate and the Lipschitz constraint is void; Dropout/DropPath are also off. Explains D logits of ±40 with BCE. Critical.
3. **Silent kill switch.** `train.py:578-584`: if measured throughput `< 0.5 it/s` (after dividing by `accumulate`) the run logs "Interrupted", saves, and `sys.exit(0)`. Any heavy arch, large batch, `accumulate ≥ 3`, or a slow first window kills training with exit code 0. The first logged window in the real run was 1.28 it/s; a 3× heavier model would have exited at iter 100. Critical ("silently stops").
4. **Gradient accumulation counter is incremented twice per iteration** (`image.py:502-505` in `closure` and `:735-738` in `optimize_parameters`). Trace for `accumulate=2`: counter goes 0→1→2→0 every iteration, so a step happens every micro-batch with loss already divided by 2 (effective LR halved, no accumulation). Even values accumulate `accum/2`, odd values work by coincidence. SAM's closure re-invocation (`fsam.py:110`) bumps it again. Critical for anyone using `accumulate`.
5. **OTF model crashes on the first batch.** `otf.py:154` reads `self.opt['jpeg_range']` (also `:146 poisson_scale_range`, `:207 poisson_scale_range2`, `:231 jpeg_range2`) from the top-level opt, but `train.py:131-132` merges `[degradations]` into `datasets.train` (every other key in the same function is read from there). `KeyError` on the very first `feed_data`. 42 of 129 templates use `model_type = "otf"`. Critical.
6. **Resume silently swaps EMA weights into the live networks and restarts the EMA.** `image.py:992-1001` saves only `net_g_ema`/`net_d_ema` under `net_g_*`/`net_d_*`; `misc.py:151-159` points `pretrain_network_*` at exactly those files; `image.py:98-108` then builds a fresh `AveragedModel` (first update copies, `n_averaged=0` **[verified]**). Raw G/D weights are lost at every resume, the schedule-free `z` state (`adan_sf.py:287`) no longer matches the parameters, GradScaler state is not saved (`base.py:462-471`), and D resumes from a smoothed D whose spectral-norm `u/v` buffers were never averaged (`use_buffers=False` default **[verified]**). High.
7. **`compile = true` produces unloadable checkpoints.** `OptimizedModule.state_dict()` keys are `_orig_mod.*` **[verified]**; `base.py:229-235` only unwraps DP/DDP, `base.py:347` only strips `module.`. Saved nets cannot be loaded into an uncompiled net (`convert.py`, `test.py`, pretrain) and `strict` failures end in `sys.exit(0)` (`base.py:445`). High.
8. **Config is parsed (argparse on `sys.argv`) and RNGs re-seeded at import time and per layer.** `utils/rng.py:11`, `archs/arch_util.py:16` (`net_opt`, called by every `DropPath.__init__`, `:211`), `optimizers/soap_sf.py:17` all call `parse_options`, which calls `set_random_seed` (`options.py:213`). `metagan` builds 17 `DropPath`s → 17 re-seeds during construction; with `manual_seed` set, every `GatedCNNBlock` after the first is initialised from the same RNG state (`metagan_arch.py:103-104`). It also makes `import neosr.data.augmentations` fail without `-opt`, and DataLoader workers (spawn) re-run it. High.
9. **AMP steps are skipped silently.** `GradScaler` is enabled for bf16 too (`image.py:149-153`); an inf/NaN gradient skips the optimizer step and halves the scale with no log **[verified: param unchanged, scale 32→16]**. The only NaN check is on the loss (`image.py:717`), so a run can stall with finite losses and nothing learning. High.
10. **Wavelet-guided GAN is internally inconsistent and slow.** After `wavelet_init` D is trained on HF-band composites (`image.py:656-674`) while G's adversarial term is evaluated on RGB (`:611`); `wavelet_guided()` is `@torch.no_grad()` (`wavelet_guided.py:386`) so it could never carry gradient; and it rebuilds `pywt.Wavelet` + `SWTForward` (+ 4 H2D copies) every iteration (`:389-400`) and pads with int64 index-gather (`mypad`, `:8-37`) instead of `F.pad(mode="circular")`. High (reached at iter 80k in the live config).

## 2. Findings table

Severity: C=critical, H=high, M=medium, L=low. Conf = confidence.

### A. Stability

| id | sev | where | defect | failure scenario | fix | conf |
|---|---|---|---|---|---|---|
| A1 | C | image.py:610-612 | D forward for G loss under `inference_mode` | no adversarial gradient; D saturates (logs) | `net_d.eval()`, params `requires_grad=False`, plain forward (no inference_mode) | verified |
| A2 | C | image.py:654, :692 | D step runs with D in eval() | spectral norm frozen, dropout off → unconstrained D | `net_d.train()` for D step, `eval()` only for G step | verified |
| A3 | C | train.py:578-584 | exit(0) when it/s < 0.5 | heavy arch / accum ≥3 / slow window kills run | delete block | high |
| A4 | C | image.py:502-505, :735-738; base.py:268 | counter incremented in both closure and optimize_parameters | even `accumulate` steps every accum/2 micro-batches with scaled loss; LR schedulers step on wrong cadence | one counter owned by the train loop, pass `apply_step: bool` into `optimize_parameters` (competitor pattern) | high |
| A5 | C | otf.py:146,154,207,231 | degradation keys read from top-level opt | KeyError on first OTF batch | read from `opt["datasets"]["train"]` (or typed schema) | high |
| A6 | H | image.py:992-1003; misc.py:151-159; image.py:98-108 | only EMA nets saved, loaded back as live nets, EMA restarted | resume ≠ continuation; SF z-state mismatch; raw weights lost | save `net_g`, `net_g_ema`, `net_d`, scaler, EMA `n_averaged` in `.state`; load EMA into the AveragedModel | high |
| A7 | H | base.py:229-235, :347; convert.py:102-106 | compiled module not unwrapped; `_orig_mod.` not stripped | compile=true → unloadable checkpoints, exit(0) | unwrap `OptimizedModule` in `get_bare_model`; canonicalise keys on load | verified |
| A8 | H | image.py:149-153, :717; base.py none | scaler enabled for bf16; no skipped-step detection | silent stall; scale→0 over time | disable scaler when dtype is bf16; log `scale`/`skipped` (compare scale before/after `update`) | verified |
| A9 | H | image.py:538-540, :656-674, :611; wavelet_guided.py:386 | D on HF bands, G on RGB, no grad | adversarial signal meaningless after wavelet_init | one differentiable `WaveletHF` transform applied to both phases (see B) | high |
| A10 | H | rng.py:11; arch_util.py:16,211; soap_sf.py:17; options.py:213 | parse_options + reseed at import/ctor | identical block inits with manual_seed; import of neosr needs `-opt`; worker re-parse | parse once in train.py; pass `scale`, `training`, `seed` explicitly | high |
| A11 | H | image.py:695-700 vs :751, :754-761; fsam.py:110 | SAM path: D backward unscaled but `gradscaler_d.step` divides by scale; `update()` skipped; closure re-run accumulates D grads twice | SAM+AMP: D frozen; SAM w/o AMP: D gets 2× gradient | with SAM, step D with plain optimizer; zero D grads before second closure or exclude D from closure | high |
| A12 | H | train.py:225-238; misc.py:62-66,70-76; base.py:363,485 | failed resume load is swallowed → `make_exp_dirs` archives the experiment dir and restarts from 0; saves are non-atomic | Ctrl-C during save → truncated `.state` → next `--auto_resume` silently restarts from scratch and moves the old run away | atomic save (tmp + `os.replace`); raise on resume failure; never rename a dir when `auto_resume` | high |
| A13 | H | image.py:98-108 (`AveragedModel(..., use_buffers=False)`) | BN running stats not averaged | archs with `nn.BatchNorm2d` (rcan, dat, hasn, sebica, msdan, dctlsa) validate/save EMA with init BN stats → garbage output when `ema>0`; D-EMA spectral `u/v` stale | `use_buffers=True` | verified default |
| A14 | M | image.py:851 then :889; base.py:340-356 | `is_train=False` set before `sf_mode` computed → SF optimizer never put in eval for validation; `save_network` copies params before `optimizer.eval()` | validation/saved weights are SF "y" iterates, not "x" | compute sf_mode first; call `optimizer.eval()` before building the state_dict; decide EMA-vs-x policy (open question) | high |
| A15 | M | augmentations.py:250-253, :287-290 | LQ bicubic up- then down-sampled unconditionally, even when "none" is drawn | every batch's LQ is re-blurred when `augmentation` is set (all *_otf templates) | early-return on "none"; resample only for mixing augs | high |
| A16 | M | img_util.py:102-103 | `n_tensor = (tensor_ - ...)` uses unclamped `tensor_` | uint8 wrap-around artifacts + wrong metrics whenever input is not already CPU fp32 (bf16/GPU) | use `n_tensor` | high |
| A17 | M | data/__init__.py:99-117 (`dataloader_args` unbound in dist branch); base.py:551 (`dist.reduce` returns None); image.py:498,646 (`requires_grad` toggling under DDP) | DDP path cannot run | NameError / TypeError on `--launcher pytorch` | fix or delete DDP/DataParallel (open question) | high |
| A18 | M | paired_dataset.py:109-141; otf_dataset.py:156,166; transforms.py:122,127 | retry loop never re-reads, `gt_path[index]` indexes a string, `randint(0, len)` off-by-one, `imfrombytes(None)` raises TypeError not AttributeError, `sys.exit` inside workers | one unreadable/mismatched image kills the run | load → on failure pick another index (bounded), raise clear error at dataset build for shape mismatch | high |
| A19 | M | image.py:462-494, :511-517 | ECO mutates `self.gt`; closure re-run under SAM blends twice; two extra forwards | ECO+SAM corrupts targets | make ECO a `prepare_batch` hook run once per iteration | medium |
| A20 | M | train.py:454,460,536; logger.py:143 | resume iter divided by `accumulate` twice; logger divides again | wrong iteration/epoch/ETA after resume or with accumulate>1 | keep one "optimizer steps" counter | high |
| A21 | M | calculate.py:201,238; topiq.py:272; dists_loss.py:107-117 | metric networks rebuilt + weights loaded per image; weights downloaded at ctor | validation takes minutes; offline runs fail | cache metric modules on the model; download once with checksum | high |
| A22 | L | consistency_loss.py:138 | dark branch `img * (img * 24389/27)` squares Y (should be `img * 24389/27`) | wrong L* for dark pixels | fix formula | high |
| A23 | L | msswd_loss.py:171-178 | `ms_swd = ms_swd + swd` outside the loop | only coarsest scale used, /num_scale | indent into loop | high |
| A24 | L | vgg_perceptual_loss.py:149-151 | `patch()` is `@torch.no_grad()` | PatchLoss/IPK terms contribute no gradient | remove decorator | high |
| A25 | L | vgg_arch.py:128-132 | `getattr(vgg, vgg_type)` returns a function; `.load_state_dict` on it | crash if `VGG_PRETRAIN_PATH` exists | call it first | high |
| A26 | L | ncc_loss.py:22-23; kl_loss.py:23-24 | `view(C, -1)` mixes batch; softmax over RGB channels | mathematically meaningless (unused in templates) | delete or fix | high |
| A27 | L | image.py:331-332 | `msg` assigned, never raised | patch_size%4 check is a no-op | raise | high |
| A28 | L | base.py:407,445; train.py exits | `sys.exit(0)` on load failures | wrappers see success | raise | high |
| A29 | L | train.py:539-548 | `break` leaves inner loop only; outer loop re-creates iterators | extra worker respawn at end | `return`/flag | high |
| A30 | L | train.py:346 | `allow_fp16_accumulation=True` under "fast_matmul" | more overflow risk with fp16 AMP | drop or gate on bf16 | medium |

### C. Performance

| id | sev | where | defect | effect | fix | conf |
|---|---|---|---|---|---|---|
| C1 | H | base.py:559 | `.item()` per loss per iteration (11 entries in the esrgan config) | ≥11 host syncs/iter; CPU launch queue drains each time | keep detached tensors; `.item()` only at `print_freq` | high |
| C2 | H | image.py:717; consistency_loss.py:189; ff_loss.py:98-107; adan*.py (`max_grad_norm`) | `isnan` on loss, `if cosim < 1e-3`, asserts with `.item()`, boolean-mask assignment | 2–5 more syncs/iter | `torch.where`/`nan_to_num`; check finiteness at print time | high |
| C3 | H | image.py:149-153 | scaler enabled for bf16 → `found_inf.item()` per optimizer step **[verified]** | 2 syncs/iter + A8 | `enabled = use_amp and dtype == float16` (default.py:114-117 had it) | verified |
| C4 | M | train.py:320; test.py:19; convert.py:298 | global `torch.set_default_device("cuda")` | TorchFunctionMode hook on every torch call (documented cost **[verified]**); CUDA sampler (`data_sampler.py:40-42`); hidden device reliance in `wavelet_guided.py:65-68,98-101`, `consistency_loss.py:42,132`, `image.py:542`, `fdl_loss.py:293` | remove; make those sites explicit buffers / `device=x.device` | high |
| C5 | M | data/__init__.py:93-98,141-142; train.py:682 | `pin_memory=False`, `persistent_workers=False`, spawn, 4 workers | `non_blocking` copies are synchronous; workers (re-importing 50+ archs) respawn every epoch — epoch boundaries show 3.5→2.9 it/s in the logs | `pin_memory=True`, `persistent_workers=True`, workers=min(8, cpu) | high |
| C6 | M | wavelet_guided.py:389-400, :8-37 | module rebuilt per call; index-gather padding | large per-iter overhead after wavelet_init | build once as a module with buffers; `F.pad(mode="circular")` | high |
| C7 | M | image.py:807,914; base.py:447-448,506-507,530-531; train.py:461 | `empty_cache`/`gc.collect` in per-image/per-tile/per-save paths | allocator thrash, stalls | remove; at most once before validation | high |
| C8 | M | fdl_loss.py:317-320 | DINOv2 ViT-B/14 forced to fp32 for both sr and gt every iteration | likely dominant cost at batch 16×256² | run backbone under autocast bf16, keep FFT/sort fp32; measure | medium |
| C9 | M | base.py:156-161; prefetch_dataloader.py:100 | nets in channels_last only when `use_amp`; inputs NCHW; losses `.to(memory_format=channels_last)` | layout conversions on every conv | decide one layout; convert in the prefetcher (`memory_format=`) | high |
| C10 | L | image.py:787-816 | tiled validation round-trips through CPU, per-tile `empty_cache` | slow val | keep tiles on GPU, pre-allocate output | high |
| C11 | L | base.py:164 | `torch.compile(mode="max-autotune", dynamic=True)` | very long compiles, recompiles on val sizes, EMA deepcopy of compiled module | `mode="default"`, compile only the training forward; validate with the bare EMA module | medium |
| C12 | L | metagan via arch_util DropPath / rng.py (A10) | argparse + env writes per layer / per import | slow startup | A10 fix | high |

Tooling tip for step 5 of the plan: `torch.cuda.set_sync_debug_mode("warn")` for 50 iterations lists every sync site with a stack trace; use it as the acceptance test for C1–C3.

### D. Structure / readability

| id | sev | where | defect | fix |
|---|---|---|---|---|
| D1 | H | models/default.py (1200), simple_otf.py, eqr_otf.py, pancam_otf.py, image.py.helaman | second, incompatible model hierarchy (keys `gt_size`, `wavelet_guided="on"/"disc"`, `color_opt/luma_opt/gw_opt` not in registry, mandatory scheduler, `pytorch_optimizer`); no template references them | delete (see §5) |
| D2 | H | base.py / image.py | one class does building, optimisation, AMP, EMA, SAM, ECO, losses, validation, saving (1000 lines); `sys.exit` ×12 in library code (base.py:199,227,377,407,445,499; image.py:310-350,373-433) | split: `Trainer` (loop, accumulation, AMP, EMA, save/resume), `SRModel` (nets+forward), `LossPipeline` (§3), `Validator`; raise `ConfigError`, exit only in train.py |
| D3 | M | base.py:358-377 vs :480-499 | duplicated retry/save | one `atomic_save()` |
| D4 | M | train.py:154 + logger.py:118 (`logger.total_iter`) vs image.py:159 (`train.total_iter`); image.py:38 (`path.print_network`) vs :46 (`print_network`); `match_lq_colors` vs `match_lq`; `warmup_iter` in micro-iters; data/__init__.py:131-139 unreachable `psutil` block | inconsistent/ dead config keys | typed strict schema (E1) |
| D5 | M | train.py:414-446, :241-268, :672-679; archs/aether_arch.py (unregistered), aether_core.py, common/ | QAT/aether/`common` path: unreachable from configs, nothing imports `common` | delete or move to its own package |
| D6 | M | train.py:235-238, 297-317, 381-386; base.py:413-424, 440-445; options.py:27-38; dists_loss.py:112-114; convert.py:91-100,121-127; img_util.py:211-215 | bare/blanket `except` swallowing errors | catch specific exceptions, re-raise with context |
| D7 | M | neosr/__init__.py star-imports; losses/data/models `__init__` import everything; archs/__init__ rescans on every `build_network` | heavy imports (also in every spawned worker) | import by name on demand |
| D8 | L | losses/loss_util.py (dup of ldl_loss:75-96); resnet_perceptual_loss.py + archs/resnet_arch.py (unused, `percep_loss` unbound); transforms.py:37-66, 163-245; arch_util.py:166-181, 234-246; img_util.py:136-154; prefetch_dataloader.py:9-65; data/otfgray_dataset.py (empty) | dead code | delete |
| D9 | L | metrics/__init__.py:1-20 duplicated block; `@staticmethod` on module functions (calculate.py:68, misc.py:169, wavelet_guided.py:106); image.py:623,688-689 `x.detach()` no-ops; train.py:654-659 double close; train.py:42-44 py3.10 check vs pyproject 3.13 | noise | clean |
| D10 | L | options/*.toml ×129 (130 lines each, 85 identical loss blocks) | template drift | generate from one base + per-arch overrides |

## 3. Loss pipeline design (B)

Principles: a loss pulls what it needs from a shared `LossContext`; the trainer runs one loop; "needs" are declared and validated at build time; existing `*_opt` keys and log keys stay.

```python
# neosr/losses/base.py
class LossContext:
    """Per-step data shared by all losses. Derived tensors are lazy and cached."""
    __slots__ = ("sr", "gt", "lq", "iter", "scale", "net_d", "net_g_ema", "_cache")
    def __init__(self, sr, gt, lq, iter, scale, net_d=None, net_g_ema=None):
        self.sr, self.gt, self.lq, self.iter, self.scale = sr, gt, lq, iter, scale
        self.net_d, self.net_g_ema, self._cache = net_d, net_g_ema, {}
    def cached(self, key, fn):
        if key not in self._cache: self._cache[key] = fn()
        return self._cache[key]
    @property
    def lq_up(self):            # bicubic-upsampled lq in [1/255, 1]
        return self.cached("lq_up", lambda: F.interpolate(self.lq, scale_factor=self.scale,
                           mode="bicubic", antialias=True).clamp_(1 / 255, 1))
    @property
    def sr_ema(self):           # EMA-generator prediction, no grad
        return self.cached("sr_ema", lambda: self.net_g_ema(self.lq).detach())

class Loss(nn.Module):
    needs: frozenset[str] = frozenset()        # subset of {"net_d", "net_g_ema"}
    def __init__(self, loss_weight=1.0, start_iter=0, target="gt"):
        super().__init__()
        self.loss_weight, self.start_iter, self.target = loss_weight, start_iter, target
    def active(self, it): return it >= self.start_iter
    def pair(self, ctx):  return ctx.sr, (ctx.gt if self.target == "gt" else ctx.lq_up)
    def forward(self, ctx) -> Tensor: raise NotImplementedError   # unweighted scalar

class PairLoss(Loss):                        # adapter: wraps every existing (pred, target) loss
    def __init__(self, fn, **kw): super().__init__(**kw); self.fn = fn
    def forward(self, ctx): return self.fn(*self.pair(ctx))

class AdversarialLoss(Loss):
    needs = frozenset({"net_d"})
    def __init__(self, gan_type="bce", view="rgb", view_start_iter=0, **kw):
        super().__init__(**kw)
        self.crit = {"bce": nn.BCEWithLogitsLoss(), "mse": nn.MSELoss(), "huber": nn.HuberLoss()}[gan_type]
        self.view = WaveletHF(view_start_iter) if view == "wavelet_hf" else nn.Identity()  # differentiable, cached filters
    def _bce(self, pred, real): return self.crit(pred, torch.full_like(pred, float(real)))
    def forward(self, ctx):                  # G term: D in eval(), params frozen, grads flow into sr
        return self._bce(ctx.net_d(self.view(ctx.sr, ctx.iter)), True)
    def d_step(self, ctx):                   # D terms: D in train(); same view on both sides
        real = ctx.net_d(self.view(ctx.gt, ctx.iter)); fake = ctx.net_d(self.view(ctx.sr.detach(), ctx.iter))
        logs = {"out_d_real": real.mean().detach(), "out_d_fake": fake.mean().detach()}
        return self._bce(real, True), self._bce(fake, False), logs

# neosr/losses/__init__.py
_ALIAS = {"pixel": "pix", "perceptual": "percep"}            # keep existing log keys
def build_losses(train_opt, *, has_d, has_ema) -> dict[str, Loss]:
    losses = {}
    for key, opt in train_opt.items():                       # TOML order == log order
        if not key.endswith("_opt") or key.startswith("optim"): continue
        name = key.removesuffix("_opt"); opt = dict(opt); cls = LOSS_REGISTRY.get(opt.pop("type"))
        base = {k: opt.pop(k) for k in ("loss_weight", "start_iter", "target") if k in opt}
        loss = cls(**opt, **base) if issubclass(cls, Loss) else PairLoss(cls(**opt), **base)
        avail = ({"net_d"} if has_d else set()) | ({"net_g_ema"} if has_ema else set())
        if loss.needs - avail: raise ConfigError(f"{key} requires {sorted(loss.needs - avail)}")
        losses[f"l_g_{_ALIAS.get(name, name)}"] = loss
    return losses

# trainer step (one uniform loop; the only structural branch is "is there a D", a model-level fact)
ctx = LossContext(sr, gt, lq, it, self.scale, self.net_d, self.net_g_ema)
logs, total = {}, sr.new_zeros(())
for key, loss in self.losses.items():
    if loss.active(it):
        v = loss.loss_weight * loss(ctx); total = total + v; logs[key] = v.detach()
logs["l_g_total"] = total.detach()
# ... G backward/step ...
for loss in self.adv_losses:                 # [l for l in losses.values() if isinstance(l, AdversarialLoss)], fixed at init
    l_real, l_fake, extra = loss.d_step(ctx); l_d = 0.5 * (l_real + l_fake)
    logs |= {"l_d_real": l_real.detach(), "l_d_fake": l_fake.detach(), "l_d_total": l_d.detach(), **extra}
```

Design notes:
- **Where data lives**: `LossContext` is created once per step by the trainer; `net_d`/`net_g_ema` are passed as references, never copied. Expensive derived tensors (`lq_up`, `sr_ema`, wavelet bands inside `WaveletHF`) are computed at most once per step because they are cached on the context.
- **Weights/schedules**: loss classes return unweighted values; the pipeline multiplies. `start_iter` on the base class replaces `wavelet_init` (as `gan_opt.view_start_iter`), `eco_init`, `sam_init`-style gating for losses. Logged value = weighted contribution (same as today, so TensorBoard histories stay comparable).
- **GAN**: one `AdversarialLoss` object owns both phases. The trainer still has a G phase and a D phase (an optimisation fact), but inside each phase the loop is uniform; D mode (`eval()` for G phase, `train()` for D phase) and `requires_grad` toggling are done by the trainer around the loops, not per loss.
- **Backward compatibility**: `*_opt` tables unchanged; `train.match_lq_colors=true` is translated by a shim into `target="lq_up"` for `consistency_opt`/`msswd_opt`; `train.wavelet_guided/wavelet_init` into `gan_opt.view="wavelet_hf", view_start_iter=N`. Shims log a deprecation line.
- **ECO, SAM, clamp** are not losses: ECO becomes a `prepare_batch(ctx)` hook in the model (run exactly once per iteration, so SAM's second closure reuses the blended targets), SAM stays an optimizer wrapper, clamp a post-process of `sr`.

Loss mapping:

| config key | class today | new | context inputs | log key | note |
|---|---|---|---|---|---|
| pixel_opt | L1Loss/MSELoss/HuberLoss/chc_loss | PairLoss | sr, gt | l_g_pix | unchanged |
| mssim_opt | mssim_loss | PairLoss | sr, gt | l_g_mssim | |
| fdl_opt | fdl_loss | PairLoss | sr, gt | l_g_fdl | consider bf16 backbone (C8) |
| consistency_opt | consistency_loss | PairLoss, target gt/lq_up | sr, gt or lq_up | l_g_consistency | fix A22; `mean` → buffer |
| msswd_opt | msswd_loss | PairLoss, target gt/lq_up | sr, gt or lq_up | l_g_msswd | fix A23 |
| perceptual_opt | vgg_perceptual_loss | PairLoss | sr, gt | l_g_percep | fix A24 |
| dists_opt | dists_loss | PairLoss | sr, gt | l_g_dists | share instance with the metric |
| ldl_opt | ldl_loss | Loss (`needs={"net_g_ema"}` when `use_ema=true`) | sr, gt, sr_ema | l_g_ldl | paper formulation uses EMA output (competitor does this) |
| ff_opt | ff_loss | PairLoss | sr, gt | l_g_ff | remove `.item()` asserts |
| ncc_opt, kl_opt | ncc_loss, kl_loss | PairLoss or delete | sr, gt | l_g_ncc/l_g_kl | A26 |
| gan_opt | gan_loss | AdversarialLoss (`needs={"net_d"}`) | sr, gt, net_d | l_g_gan, l_d_real, l_d_fake, l_d_total, out_d_real/fake | fixes A1/A2/A9 |
| train.wavelet_guided | model flag | `gan_opt.view="wavelet_hf"` | inside AdversarialLoss | — | shim |
| train.match_lq_colors | model flag | `target="lq_up"` on consistency/msswd | — | — | shim |
| ResNetPerceptualLoss | resnet_perceptual_loss | delete | | | unused, broken |

## 4. Competitor ideas worth porting (E), ranked by value/cost

1. **Strict typed config schema** (`traiNNer/utils/redux_options.py`, `msgspec.Struct` with `forbid_unknown_fields=True`, decoded in `utils/options.py:91`). Would have caught A5, D4 and most "fix import error" fallout at parse time. Value very high, cost medium (one schema module; `msgspec.toml` exists).
2. **Save raw and EMA nets separately + scaler + EMA step in the state** (`models/sr_model.py:1192-1222`, `models/base_model.py:725-748, 788-830`). Direct fix for A6. Low cost.
3. **Compiled-module unwrapping and state-dict canonicalisation** (`base_model.py:318-322`, `:644-690`). Fix for A7. Low cost.
4. **Accumulation owned by the train loop** (`train.py:596-619` passes `current_accum_iter`, `apply_gradient` into `optimize_parameters`; `train.py:994-996` discards a partial accumulation on interrupt). Fix for A4/A20. Low cost.
5. **Loss activation/weight schedule as config fields** (`losses/iterative_loss_wrapper.py`: `start_iter`, `target_iter`, `schedule_type`, `is_active`). Port the semantics as fields on `Loss` (§3), not the wrapper/`hasattr` plumbing. Replaces `wavelet_init`/`eco_init`/ad-hoc gating. Low–medium cost.
6. **LDL with EMA output** (`losses/ldl_loss.py:84`, `sr_model.py:579-596`). Falls out of §3 for free.
7. **Skipped-step and grad-norm logging** (`sr_model.py` ~`:722-731`: `scale_before/after` → `optimizers_skipped`, `grad_norm_g`). Fix for A8 observability. Trivial.
8. **pytest suite** (`tests/test_losses/test_losses.py`: every loss on black/white/random pairs, finite, zero on identical inputs, dtype/device matrix; `tests/test_models`, `tests/test_data`). The refactor needs this safety net first. Medium cost.
9. **Loader defaults** `pin_memory=True`, `persistent_workers=True`, CUDA prefetcher refuses to run without pinned memory (`redux_options.py:69-72`, `train.py:532-533`). Trivial.
10. **Crash handling** (`train.py:984-1005`: `logger.exception`, still save, exit) and `save_checkpoint_format="safetensors"` (`base_model.py:437-447`). Low cost, nice-to-have.

Not worth porting: training-automation manager, dynamic loss scheduling (905 lines), VRAM manager/OOM batch-resizing, `WorkerHealthMonitor`, the R3GAN/contrastive `isinstance` special cases inside the loss loop (the thing the owner rightly dislikes), pixel-format conversions.

## 5. Deletion list

Files (no template or import references outside the group, checked by grep):
- `neosr/models/default.py`, `simple_otf.py`, `eqr_otf.py`, `pancam_otf.py`, `image.py.helaman` — parallel hierarchy with incompatible keys and unregistered losses; `model_type` in all 129 templates is `image` (87) or `otf` (42).
- `neosr/data/pcpaired_dataset.py`, `pancamotf_dataset.py`, `pancamsingle_dataset.py`, `eqrotf_dataset.py`, `simple_otf_dataset.py`, `otfgray_dataset.py` (empty) — only used by the deleted models.
- `neosr/losses/loss_util.py` (duplicate of `ldl_loss.get_refined_artifact_map`, only `default.py` imports it), `neosr/losses/resnet_perceptual_loss.py`, `neosr/archs/resnet_arch.py`.
- `neosr/utils/rng.py` and `arch_util.net_opt`/`soap_sf.toml_opt` after A10 (seed/scale passed explicitly).
- `common/` + `neosr/archs/aether_arch.py`, `aether_core.py` + train.py QAT block (`train.py:414-446`), `generate_calib_data` (`:241-268`), `sys.path` hack (`:672-679`), `image._init_ema` — unreachable from configs (aether is unregistered, nothing imports `common`). If aether matters, give it its own package with a registered arch.
- `neosr/data/prefetch_dataloader.py:9-65` (`PrefetchGenerator`, `PrefetchDataLoader`; only `CUDAPrefetcher` is used).

Functions/blocks:
- `train.py:578-584` kill switch; `:297-317` nvcc check; `:42-44` py3.10 check; `:654-659` duplicate close; `:403-412` MACs block unless kept behind an explicit flag.
- `transforms.py:37-66` `single_random_crop`, `:163-245` `paired_random_crop_ow`; `arch_util.py:166-181` `pixel_unshuffle_width`, `:234-246` `normal_init/constant_init`; `img_util.py:136-154` `tensor2img_fast`; `data/__init__.py:131-139` unreachable block; `metrics/__init__.py:1-20` duplicated imports; `image.py:623,688-689` no-op `detach()`s; `image.py:159` `self.total_iter`; `image.py:103-108,998-1001` discriminator EMA (saving a smoothed D and resuming from it has no clear benefit — open question).
- Options: `train.wavelet_guided/wavelet_init`, `train.match_lq_colors` (become loss fields via shim), `train.eco_*` → one `[train.eco]` table, `logger.total_iter` → `train.total_iter` (shim for one release), `ncc_opt`/`kl_opt` (A26).

## 6. Refactor plan

| step | content | parallel? | verification |
|---|---|---|---|
| 0 | Safety net: `tests/` with (a) every loss on random/identical pairs: finite, >0, gradient reaches `pred`; (b) all 129 templates parse; (c) 20-iteration smoke on a tiny arch for `image` and `otf`; (d) save→resume equality of all parameters and optimizer state. | — (first) | `pytest -x`; (d) must initially **fail** for A6 |
| 1 | Hotfixes on the current structure, one PR: A1, A2, A3, A4 (single counter), A5, A8 (scaler off for bf16 + skipped-step log), A13 (`use_buffers=True`), A14, A15, A16, C1–C3. | yes, with step 2 (disjoint files except image.py; do 1 first) | 2k-iter run of `train_esrgan2.toml`: `out_d_*` bounded (|logit| < ~10), `l_d_total` not → 0, `l_g_gan` moves other losses; it/s ≥ before; `set_sync_debug_mode("warn")` shows ≤3 syncs/iter |
| 2 | Checkpoint/resume: save raw+EMA+D+scaler+EMA step, atomic save, compile unwrap, resume failure raises (A6, A7, A12, D3), remove `sys.exit` from base.py (D2 part). | yes, with step 3 | test 0(d) passes; `compile=true` round trip through `convert.py`; kill -INT during save leaves a loadable previous state |
| 3 | Loss pipeline (§3): `losses/base.py`, `build_losses`, `WaveletHF` module, LDL-EMA, shims; `image.closure` becomes the uniform loop. | yes, with step 2 | equivalence test: old vs new `l_g_total` on a fixed batch equal to 1e-6 (GAN term excluded, it was dead); identical log keys; unit tests for `needs` validation |
| 4 | Deletions (§5) + import-time parsing removal (A10) + typed schema (E1) + key unification (D4). | after 3 merges | `python -c "import neosr"` with no argv; templates parse under strict schema; ruff clean; model build with `manual_seed` gives distinct block inits |
| 5 | Performance: C4–C11 (default device removal, loader defaults, channels_last decision, wavelet cache, metric cache, no `empty_cache`, compile mode). | yes, with 4 (different files) | `torch.profiler` trace before/after; it/s on esrgan + otf configs; sync-debug count; validation wall time |
| 6 | Structure: Trainer/SRModel/Validator split, DDP fix-or-remove decision, template generation (D10), readme update. | after 4/5 | full 2k-iter runs image + otf; resume mid-run; `test.py` + `convert.py` on the resulting checkpoints |

## 7. Open questions / not verified

1. **Which symptom is "stability problems"?** The logs show (a) D collapse with a gradient-less G (A1/A2), (b) runs ending with "Interrupted" (A3 would print exactly that), and (c) resumes that swap EMA weights (A6). All three are consistent with "unstable"; confirm which the owner saw.
2. **Wavelet-guided intent**: should D only see HF bands (current), or should G's adversarial term also go through the bands (my proposal, needed for any gradient)? I could not compare against upstream neosr.
3. **Schedule-free + EMA policy**: validate/save EMA(y) (today), SF's x, or EMA(x)? The three differ; the code currently does none of them consistently (A14).
4. **Discriminator EMA**: keep saving/resuming a smoothed D?
5. **DDP/DataParallel**: currently unusable (A17). Fix or drop multi-GPU?
6. **aether/QAT/common, pancam/eqr archs**: dead from the config side; delete or keep as separate packages?
7. **Logging convention**: keep logging weighted contributions (current) or switch to raw values?
8. Not measured: actual throughput impact of C1–C9 (needs a profiler run — left to the training agent) and quality impact of running the DINOv2 backbone in bf16 (C8).
9. The `spectral_norm` claim was verified behaviourally (eval() leaves `weight_u` unchanged); the source grep for `do_power_iteration=module.training` failed only because of line wrapping.
