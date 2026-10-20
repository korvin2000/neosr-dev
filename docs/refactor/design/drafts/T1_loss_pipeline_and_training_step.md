# T1 — Loss pipeline and training step (design, v2 after Codex review)

Repo: /home/korvin/apps/neosr-dev (read-only). Runnable skeleton + GPU probe: `scratchpad/design/skeleton/` — `losses_base.py` (85 lines), `gan.py` (75), `phase.py` (96), `step.py` (149), `probe.py`; **36/36 checks pass** on torch 2.9.1+cu128 / RTX 4090. Codex (gpt-6.1-sol, xhigh, read-only) reviewed v1: `codex_T1_review.md`; §8 lists what was adopted, rejected, deferred.

## 0. Decisions (one-line rationale each)

| # | Decision | Rationale |
|---|---|---|
| D1 | Every objective term is a `Loss(nn.Module)` with class attrs `inputs`, `phase`, `log_key`, `fp32`; the trainer loop is `total = sum(t.compute(ctx) for t in terms)` | Zero branching in the loop; each loss declares what it pulls from the context (owner's requirement). |
| D2 | `compute(ctx)` is the one wrapper (precision policy, `loss_weight` applied once, logging); kernels stay in `forward(*tensors)` unweighted; model-using terms override `evaluate(ctx)`, never `compute` | Weight/precision/logging cannot be bypassed by a subclass (Codex #9); existing two-tensor kernels stay testable. Today 15 forwards multiply the weight themselves and `ResNetPerceptualLoss` drops it. |
| D3 | `Context` per micro-batch: `sr, gt, lq, it, scale, net_d, net_g_ema`, `cached_property` `lq_up` (fp32) and `sr_ema`, plus `cache` dict for tensors terms share | Derived inputs computed at most once; no model-wide `self.lq_interp` state (image.py:524-535). |
| D4 | `gan_opt` builds one `gan_loss`; `expand()` returns `[gan_loss (phase g), gan_d_loss (phase d)]` sharing criterion and adversarial view | GAN G/D terms are ordinary terms of two phases; nothing in the trainer knows "gan". |
| D5 | Adversarial view (identity, or differentiable wavelet HF bands from `wavelet_init`) is one function applied to `sr` (G term, with grad, cached), `gt` and the cached `sr` view detached (D term) | Fixes the domain mismatch (D on HF bands, G in RGB: image.py:611 vs :657) and the no-grad view (wavelet_guided.py:386). |
| D6 | G pass: the D parameters that were trainable at construction get `requires_grad_(False)`; D never leaves `train()`; D pass restores the same set | Adversarial gradient reaches G (today `inference_mode`, image.py:610); matches BasicSR/Real-ESRGAN; preserves intentional freezes (Codex #5). |
| D7 | One `Phase` per network (terms, optimizer, scaler, clip, schedulers, on-device non-finite counter); G and D are two instances | Removes the G/D copy-paste (image.py:625-714) and the fsam closure entanglement. |
| D8 | One micro counter; `optimize_parameters(it) -> bool` ("window closed, update attempted"); train.py counts attempted updates from it; `it` = completed updates + 1 | Fixes the double increment (image.py:502+735) and the accumulate re-divisions (train.py:454,600; logger.py:143,153); keeps today's 1-based `wavelet_init` threshold (Codex #12). |
| D9 | GradScaler only for fp16 (one per phase); bf16/fp32 unscaled | Scaler on bf16 is dead weight (image.py:149-153) plus a host sync per update; separate backward graphs → separate scales. |
| D10 | `finite` = on-device result of `torch._amp_foreach_non_finite_check_and_unscale_` (the op GradScaler uses); counters read at log windows and before save/validation/shutdown; abort only for an *unprotected* phase or a fully skipped window | No per-update host sync. Verdict equals the scaler's (probe 9: `isfinite(grad norm)` mis-flags a finite 1e25 gradient). bf16 EMA cannot be shielded sync-free (lerp propagates NaN, probe 4a) → detect-and-abort, opt-in `strict_finite`. |
| D11 | 20-line `Ema` (deepcopy of the bare module, `_foreach_lerp_` gated by `finite`) instead of `AveragedModel` | Sync-free gate for fp16/strict skips, no `n_averaged`, no deepcopy of compiled/DDP wrappers (base.py:163-175, image.py:98). |
| D12 | Logs: detached GPU scalars summed on device; `get_current_log()` = one packed `.tolist()` per `print_freq` window; `all_reduce` (collective on every rank) with attempted-update count reduced in the same units | Removes `.item()` per loss per iteration (base.py:559) and the broken `distributed.reduce` (base.py:551); fixes the rank-unit mismatch (Codex #11). |
| D13 | `match_lq_colors` → per-loss `target = "lq_up"`; legacy flag mapped for `consistency_opt`, `msswd_opt` | Same semantics as image.py:572-585, no global flag, no branching. |
| D14 | Delete SAM/fsam, D-EMA, `warmup_iter`, `update_loss_weights`/`get_current_gan_weight`, the throughput kill switch | Zero of 135 configs use SAM, `warmup_iter` or a scheduler; D-EMA corrupts resume (saved as net_d, image.py:999); hooks exist only for default.py. |
| D15 | Keep ECO as a 25-line pure transform `(lq, gt) = eco(net_g, lq, gt, it)` | Isolated, documented upstream feature, no longer mutates `self.gt` (image.py:477). Zero configs → deletable at no cost. |
| D16 | Schedule-free: `eval_weights(swap_sf)` context (optimizer `.eval()` only when the *live* net is evaluated; module eval; restored in `finally`); EMA keeps averaging the live y-sequence | Fixes the never-taken SF switch (image.py:851 vs :889), the CPU copy before `.eval()` (base.py:343 vs :353), the EMA-flipped-to-train bug (image.py:899); tracking x would be an algorithm change (Codex #7). |
| D17 | Generic `[[train.losses]]` list is phase 2, after the legacy builder is stable | Codex #13; nothing in the trainer changes for it. |

## 1. Loss pipeline

### 1.1 Context (`neosr/losses/base.py`, skeleton `losses_base.py`)
```python
class Context:
    def __init__(self, *, sr, gt, lq, it, scale, net_d=None, net_g_ema=None): ...
        self.logs: dict[str, Tensor] = {}     # detached GPU scalars
        self.cache: dict[str, Tensor] = {}    # tensors terms share within this forward
    @cached_property
    def lq_up(self):                          # fp32 bicubic upsample, clamp(1/255, 1)
    @cached_property
    def sr_ema(self):                         # no_grad EMA generator prediction
    def log(self, key, value): self.logs[key] = value.detach()
```
`it` = 1-based index of the update being formed (today's `current_iter` for `accumulate=1`). Derived inputs are computed with autocast off (Codex #9).

### 1.2 Loss base
```python
class Loss(nn.Module):
    inputs = ("sr", "gt"); phase = "g"; log_key = ""; fp32 = False
    def __init__(self, loss_weight=1.0): ...
    def expand(self): return [self]
    def evaluate(self, ctx):                              # unweighted; default = kernel on declared inputs
        args = [getattr(ctx, n) for n in self.inputs]
        if self.fp32: args = [a.float() for a in args]
        return self(*args)
    def compute(self, ctx):                               # the single wrapper
        with torch.autocast("cuda", enabled=False) if self.fp32 else nullcontext():
            value = self.evaluate(ctx) * self.loss_weight
        ctx.log(self.log_key, value); return value
```
A loss needing the EMA generator declares `inputs = ("sr", "gt", "sr_ema")`; one needing a model overrides `evaluate` and reads `ctx.net_d`.

### 1.3 Construction (`neosr/losses/__init__.py: build_losses(train_opt, device) -> (g_terms, d_terms)`)
1. Legacy tables in today's evaluation order (image.py:546-614), today's keys: `pixel_opt→l_g_pix, mssim_opt→l_g_mssim, ncc_opt→l_g_ncc, kl_opt→l_g_kl, fdl_opt→l_g_fdl, consistency_opt→l_g_consistency, msswd_opt→l_g_msswd, perceptual_opt→l_g_percep, dists_opt→l_g_dists, ldl_opt→l_g_ldl, ff_opt→l_g_ff, gan_opt→l_g_gan`.
2. `match_lq_colors = true` (1 config) → `target = "lq_up"` injected into `consistency_opt` and `msswd_opt`.
3. `wavelet_guided = true` + `wavelet_init` (82 configs) → `gan_opt.wavelet_init`, builder passes `view=WaveletHF()`; `wavelet_guided` without `gan_opt` → `ValueError` (today sys.exit, image.py:333-336).
4. Per spec: `loss = LOSS_REGISTRY.get(type)(**opts)`; `loss.log_key = ...`; `if target: loss.inputs = (loss.inputs[0], target)`; `loss.requires_grad_(False).eval().to(device, memory_format=channels_last)`; `terms += loss.expand()`. Partition by `phase`.
5. Validation once, in the builder (raise, never `sys.exit`): ≥1 g-term (image.py:311-314); `gan_opt` ⇔ `network_d` ⇔ `optim_d` (image.py:315-330); unique log keys; `accumulate ≥ 1`; `patch_size % 4` with augmentation (the message at image.py:331-332 is never raised today); legacy `gw_opt/color_opt/luma_opt` (default.py:176-189 only) → "removed, use consistency_opt".
6. Freezing is generic: `requires_grad_(False)` covers DISTS `alpha/beta` (dists_loss.py:97-100), VGG/ResNet/DINO backbones and the SWT filters (registered as `nn.Parameter`, verified).

### 1.4 GAN terms (skeleton `gan.py`)
```python
class gan_loss(Loss):                      # phase g, l_g_gan
    def adv(self, ctx, x): return self.view(x) if ctx.it >= self.view_from else x
    def term(self, logits, real): return self.crit(logits, torch.full_like(logits, self.real_val if real else self.fake_val))
    def evaluate(self, ctx):
        adv_sr = ctx.cache["adv_sr"] = self.adv(ctx, ctx.sr)
        return self.term(ctx.net_d(adv_sr), real=True)
    def expand(self): return [self, gan_d_loss(self)]
class gan_d_loss(Loss):                    # phase d, unweighted (gan_loss.py:83-84), log_key l_d_total
    def evaluate(self, ctx):
        adv_sr = ctx.cache.get("adv_sr"); fake_in = adv_sr.detach() if adv_sr is not None else self.g.adv(ctx, ctx.sr.detach())
        real = ctx.net_d(self.g.adv(ctx, ctx.gt)); fake = ctx.net_d(fake_in)
        l_real, l_fake = self.g.term(real, True), self.g.term(fake, False)
        ctx.logs.update(l_d_real=..., l_d_fake=..., out_d_real=real.detach().mean(), out_d_fake=fake.detach().mean())
        return l_real + l_fake             # today's optimisation scale (two backward calls, image.py:699-700)
```
`WaveletHF(nn.Module)`: wavelet_guided.py:386-432 as a module built once (sym19 filters, `SWTForward` owned, cast to the input dtype), `forward(x) -> cat(LH*0.035, HL*0.025, HH*0.02)` of Y, **no** `@torch.no_grad`. Verified: gradient flows through `SWTForward`; the per-call rebuild costs ~0.2 ms, correctness (grad, dtype) is the reason to cache.

### 1.5 Mapping of every current loss

| Loss (forward) | `inputs` | weight today | change | key |
|---|---|---|---|---|
| `L1Loss`,`MSELoss`,`HuberLoss`,`chc_loss` basic_loss.py:45,77,114,180 | `(sr, gt)` | inside (:53,85,127,219) | inherit `Loss`, drop `self.loss_weight *` | `l_g_pix` |
| `mssim_loss` ssim_loss.py:112 | `(sr, gt)`, `fp32` | inside (:126) | drop weight | `l_g_mssim` |
| `ncc_loss` ncc_loss.py:36 | `(sr, gt)`, `fp32` | inside (:38) | drop weight (numerics: T-losses) | `l_g_ncc` |
| `kl_loss` kl_loss.py:21 | `(sr, gt)` | inside (:31) | drop weight | `l_g_kl` |
| `fdl_loss` fdl_loss.py:318 | `(sr, gt)` | inside (:341) | drop weight; backbone frozen by builder | `l_g_fdl` |
| `consistency_loss` consistency_loss.py:138 | `(sr, gt|lq_up)`, `fp32` | inside (:192) | drop weight; `self.mean` → buffer; `if cosim < 1e-3` (:189) → tensor select (hidden sync, Codex #10) | `l_g_consistency` |
| `msswd_loss` msswd_loss.py:166 | `(sr, gt|lq_up)` | inside (:182) | drop weight; `self.rand` → buffer (re-`.to()` per call, :149) | `l_g_msswd` |
| `vgg_perceptual_loss` vgg_perceptual_loss.py:185 | `(sr, gt)` | inside (:223) | drop weight; `scale` ctor arg instead of `net_opt()` (:16) | `l_g_percep` |
| `ResNetPerceptualLoss` resnet_perceptual_loss.py:59 | `(sr, gt)` | `perceptual_weight` only (:88) | `loss_weight` → base, drop `perceptual_weight`; frozen/eval by builder | `l_g_percep` |
| `dists_loss` dists_loss.py:144 | `(sr, gt)` | inside (:171) | drop weight; `alpha/beta` frozen by builder | `l_g_dists` |
| `ldl_loss` ldl_loss.py:99 | `(sr, gt)`, `fp32` (an EMA-teacher variant `(sr, gt, sr_ema)` is a *new* objective, not a migration) | inside (:103) | drop weight; stop storing `self.output/self.gt` (:100-101) | `l_g_ldl` |
| `ff_loss` ff_loss.py:118 | `(sr, gt)` | inside (:147) | drop weight; `.item()` asserts at :102 → diagnostics only (Codex #10) | `l_g_ff` |
| `gan_loss` gan_loss.py:64 | `ctx.net_d`, `sr`, `gt`, view | G only (:84) | rewrite per 1.4 | `l_g_gan`,`l_d_*`,`out_d_*` |
| `wavelet_guided` wavelet_guided.py:386 | — | — | `WaveletHF` view owned by `gan_loss` | — |
| `loss_util.get_refined_artifact_map` | — | — | delete (duplicate of ldl_loss.py:77-97, no caller after default.py) | — |
| default.py `gw_opt/color_opt/luma_opt` | — | — | unregistered; builder raises | — |

### 1.6 Log keys
Unchanged: all keys above, `l_g_total` (sum of weighted g-terms incl. `l_g_gan`), `l_d_total` (= `l_d_real + l_d_fake`; today's log halves it, image.py:686). New: `grad_norm_g/d`, `skipped_g/d` (no `l_` prefix → tensorboard root, like `out_d_*`). Values are window means of undivided micro-batch objectives (today: last iteration, D terms pre-divided by `accumulate`, image.py:665,680). `gan_weight` (train.py:571) goes with the legacy hook.

## 2. Training step

### 2.1 Phase (`neosr/models/phase.py`, skeleton `phase.py`)
```python
class Phase:
    def __init__(self, net, optimizer, terms, *, log_key, amp_dtype, use_amp, grad_clip=1.0, accum=1, schedulers=(), strict_finite=False):
        fp16 = use_amp and amp_dtype is torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=fp16, init_scale=2.0**5)
        self.strict = strict_finite and not fp16            # never bypass the scaler's update()
        self.protected = fp16 or self.strict
        self.nonfinite = torch.zeros((), dtype=torch.long, device=dev)
    def backward(self, ctx):                                 # DDP: net.no_sync() on non-boundary micro-batches
        with torch.autocast(**self.autocast): total = sum(t.compute(ctx) for t in self.terms)
        ctx.log(self.log_key, total)
        self.scaler.scale(total * self.inv_accum).backward()
    def step(self) -> Tensor:                                # GPU bool, no host sync
        self.scaler.unscale_(self.opt)
        grads = [p.grad for p in self.params if p.grad is not None]
        found_inf = torch.zeros((), device=dev)
        torch._amp_foreach_non_finite_check_and_unscale_(grads, found_inf, self._one)
        finite = found_inf == 0
        if self.grad_clip: norm = clip_grad_norm_(self.params, self.grad_clip, foreach=True)
        else: norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))   # measure only: no inf/inf
        if self.strict and not finite.item(): self.opt.zero_grad(set_to_none=True)
        else: self.scaler.step(self.opt); self.scaler.update(); self.opt.zero_grad(set_to_none=True)
        for s in self.schedulers: s.step()                   # clock = attempted update, skipped or not
        self.nonfinite += ~finite; self.grad_norm.copy_(norm); return finite
```
- `grad_clip`: config `true` → 1.0 (today's constant), `false` → 0, float → that norm. With clipping off the norm is measured without `clip_grad_norm_(max_norm=inf)` because `inf/inf` would inject NaN after the finite check (Codex #1, probe 11).
- Schedulers (none in any config; `MultiStepLR`/`CosineAnnealing` kept) step per attempted update inside the phase; `base.update_learning_rate`, `_get_init_lr`, `_set_lr`, `warmup_iter` go (`initial_lr` crashes without a scheduler, base.py:253; schedule-free optimizers carry `warmup_steps`).

### 2.2 The step (`image.optimize_parameters`, skeleton `step.py`)
```python
def optimize_parameters(self, it: int) -> bool:
    lq, gt = self.lq, self.gt
    if self.eco is not None: lq, gt = self.eco(self.net_g, lq, gt, it)
    for p in self.d_params: p.requires_grad_(False)          # d_params = trainable-at-construction mask
    with torch.autocast(**self.g.autocast):
        sr = self.net_g(lq)
        if self.clamp: sr = sr.clamp(1/255, 1)
    ctx = Context(sr=sr, gt=gt, lq=lq, it=it, scale=self.scale, net_d=self.net_d, net_g_ema=self.ema.module if self.ema else None)
    self.g.backward(ctx)
    if self.d is not None:
        for p in self.d_params: p.requires_grad_(True)       # D never leaves train()
        self.d.backward(ctx)
    self.micro += 1
    stepped = self.micro % self.accum == 0
    if stepped:
        finite = self.g.step()
        if self.d is not None: self.d.step()
        if self.ema is not None: self.ema.update(finite)
        self._updates += 1
    accumulate ctx.logs into on-device sums; self._n += 1
    return stepped
```
`otf.feed_data` is unchanged (it only sets `self.lq/self.gt`); `image.feed_data` keeps augmentation. Construction order in `image.__init__`: build → (QAT prepare, if ever revived: train.py:414 today runs it *after* optimizers; Codex #8) → load pretrained → `Ema(bare)` → compile/DDP wrap → phases. All cached parameter lists (`Phase.params`, `Ema.src`, `d_params`) are built last.

### 2.3 AMP, NaN/inf, "successful update" (probe-verified)
- fp16: GradScaler per phase; `scaler.step` skips on inf (its `.item()` sync is inherent; with `torch.optim.AdamW(fused=True)` the scaler hands `found_inf` to the kernel and even that sync disappears). Our `finite` is the same elementwise op → identical verdict (probe 9). EMA gate is exact: live weights untouched → `lerp` weight 0 is a true no-op (probe 5). Logged as `skipped_g`, never fatal. `strict_finite` is ignored under fp16 (it would leave the scaler `UNSCALED`, Codex #2; probe 10).
- bf16 / fp32 (85 configs): no scaler, no per-update sync. A non-finite gradient reaches the weights and — verified — `_foreach_lerp_(ema, live, 0)` still yields NaN. Policy: count on device; `check_finite()` (counter-only transfer, keeps the window) runs at every `print_freq`, **before every save/validation and at shutdown/interrupt**; it raises naming the phase, so no corrupted checkpoint is ever written. Opt-in `train.strict_finite = true`: one `.item()` per update, grads zeroed, update skipped, live + EMA clean (probe 4d-f).
- "every attempted update in the window skipped" also raises (dead run), with skipped and attempted counts reduced across ranks in the same units.

### 2.4 EMA (`neosr/models/ema.py`)
`Ema(bare, decay)`: `deepcopy(bare).eval().requires_grad_(False)`; `update(finite)` = `_foreach_lerp_(dst, src, (1-decay)*finite)` + `_foreach_copy_` of buffers. `src` holds the live `Parameter` objects; `adamw_sf.py:84,96,174`, `adan_sf.py:64` and `soap_sf.py` update in place (`lerp_`, `_foreach_*`, `set_`), so references stay valid; any future structural mutation (QAT) must rebuild `Ema`, `Phase.params`, `d_params` together. D-EMA deleted. Checkpoint keys (`params`/`params_ema`, scalers, counters) are the checkpoint topic; this class exposes `module.state_dict()` and the contract "save only after `stepped`" (interrupt saves included: train.py:628,635 must happen between windows or drop the partial window — resume restarts `micro = 0` with grads discarded, which is exactly a dropped partial window).

### 2.5 Logging and DDP
`get_current_log()`: `check_finite()` then one packed tensor `[window sums / n, grad_norm_g, grad_norm_d, nonfinite_g, nonfinite_d, attempted]`, `all_reduce` when initialized (means divided by world size, counts kept as sums), one `.tolist()`, slice, reset. Every rank calls it (collective); rank 0 prints. `MessageLogger` keeps its float-dict interface minus `// accumulate` (logger.py:143,153).

### 2.6 Schedule-free, SAM, ECO
- Schedule-free (170 tables): `with model.eval_weights(swap_sf=self.ema is None): validate/export`. Converting optimizers is only needed when the live net is evaluated; the EMA module is independent of the y/x swap. EMA(y) stays (every existing EMA checkpoint is EMA(y)); tracking x is an algorithm change, not done silently.
- SAM (fsam): **delete**. Zero configs; incompatible with accumulation (image.py:307-310), AMP (:301-306) and schedule-free; its second base optimizer is never checkpointed (image.py:405-413); it forced the stateful `closure()`. `g.backward(ctx)` is now a pure function of `(lq, gt)`, so a future `SamPhase.step()` is ~15 lines.
- ECO: **keep** (`neosr/models/eco.py`, skeleton `Eco`): pure, once per micro-batch, `clamp` on the centroid as today (image.py:516-517); it still runs one extra no_grad forward (inherent to the method).

### 2.7 train.py loop contract
```python
update = resume_state["iter"] if resume_state else 0           # attempted updates (window closes)
for epoch in ...:
    for batch in prefetcher:
        model.feed_data(batch)
        if not model.optimize_parameters(update + 1): continue   # micro-batch accumulated
        update += 1
        if update % print_freq == 0: msg_logger({"epoch": epoch, "iter": update, "lrs": model.lrs(), "time": ..., **model.get_current_log()})
        if update % save_freq == 0: model.save(epoch, update)     # save() calls check_finite() first
        if val and update % val_freq == 0: model.validation(..., update)   # idem
        if update >= total_iter: break
# finally / KeyboardInterrupt: model.check_finite(); model.save(...)
```
`iter` in state files = attempted updates; `micro` restarts at 0 on resume. `total_iter`: one source (`train.total_iter`, `logger.total_iter` accepted as fallback for the 91 configs that use it). Dropped: kill switch (train.py:577-584), `update_learning_rate`/`update_loss_weights` calls (:553-559), `gan_weight` log var (:571).

## 3. Skeletons
`scratchpad/design/skeleton/{losses_base,gan,phase,step}.py` are the runnable versions of §1.1-1.4 and §2.1-2.2 (core 85+75 lines; phase+ema 96; step incl. Eco/Trainer 149). `probe.py` exercises them with tiny nets (conv G with PixelShuffle, spectral-norm D, a high-pass stand-in for `WaveletHF`).

## 4. Migration (ordered; rows with the same letter touch disjoint files and run in parallel)

| Step | Files | Work | Verify |
|---|---|---|---|
| A1 | `neosr/losses/base.py` (new), `neosr/losses/__init__.py` | `Context`, `Loss`, `build_losses` with legacy map + validation (§1.3) | Dry-run over all 135 `options/*.toml` (parse only; mock `LOSS_REGISTRY.get` to record `(type, log_key, inputs, phase)`); assert order and keys equal today's `if self.cri_*` order; `match_lq_colors` config yields `target="lq_up"` on exactly consistency/msswd |
| A2 | `neosr/losses/gan_loss.py`, `neosr/losses/wavelet_guided.py` | `gan_loss`, `gan_d_loss`, `WaveletHF` (built once, differentiable, dtype-cast) | probe checks 1, 2, 7 against the real `WaveletHF` (grad through SWT, 3-channel output, switch at `wavelet_init`) |
| A3 | `neosr/models/phase.py`, `neosr/models/ema.py`, `neosr/models/eco.py` (new) | as skeleton | `probe.py` (36 checks) against the real modules in fp32/bf16/fp16 |
| B1 (after A1) | each `neosr/losses/*_loss.py`, one worker per file | inherit `Loss`, drop internal weight, `log_key`/`fp32`, buffers, no stored state, tensor-select for `cosim` threshold; numerics fixes stay in the losses topic | per file `allclose(new(x,y)*w, old(x,y))`; `grep -n loss_weight neosr/losses` shows only `base.py` + ctor assignments |
| B2 (after A1-A3) | `neosr/models/image.py`, `neosr/models/base.py` | replace image.py:88-351 with phase construction (≈60 lines), image.py:462-770 with §2.2; delete `reduce_loss_dict`, `update_learning_rate`, `_get_init_lr`, `_set_lr`, `update_loss_weights`, `get_current_gan_weight`; `eval_weights()` in `nondist_validation` and `save`; `check_finite()` in `save`/`validation` | 300 updates of `options/train_esrgan2.toml` (esrgan + metagan, bf16, adan_sf, 5 losses): log key set = today's + `grad_norm_*`/`skipped_*`; `out_d_real/out_d_fake` bounded (today ±40 by 1.2k iterations, experiments/train_esrgan logs); a hook on net_g shows non-zero gradient from `l_g_gan` alone; metagan dropout/droppath now active in the G pass — compare loss curves against the old run |
| B3 (after A3) | `train.py`, `neosr/utils/logger.py` | §2.7 loop; remove accumulate divisions and kill switch; `check_finite` on interrupt | `accumulate=1` vs `2` with `batch_size` halved: same update count, same save/val cadence, same `iter` in state files; resume at a boundary continues the same `iter` |
| C (after B2) | delete `models/default.py`, `simple_otf.py`, `eqr_otf.py`, `pancam_otf.py`, `image.py.helaman`, `optimizers/fsam.py`, `losses/loss_util.py`; `models/__init__.py` registry | — | `python -c "import neosr.models"` + the 135-config dry-run; `grep -rn "fsam\|loss_util\|models.default" neosr` empty |
| D (after B2) | `tests/test_step.py` | `probe.py` as pytest (CPU variant + CUDA) | run on every later change |
| E (after B2, optional) | two-rank DDP smoke, `compile = true` smoke | `no_sync` on non-boundary micro-batches; `TORCH_LOGS=recompiles` | two cached D graphs at most (frozen/unfrozen); no reducer errors with SN D; BN-based `patchgan` under DDP flagged unsupported until this passes |

## 5. Risks
1. Dynamics change on purpose: G receives adversarial gradient; D trains in train mode, so MetaGAN's dropout 0.2 and stochastic depth (metagan_arch.py:40-43,103) are active in the G pass too (today's eval passes suppressed them, Codex #5). `gan_opt.loss_weight` (0.3) was tuned against a dead GAN term; expect re-tuning. HF-band adversarial gradient on the G side is new (literature-consistent, unvalidated here).
2. bf16 non-finite updates corrupt live and EMA weights until the next check (≤ `print_freq` updates); never a checkpoint. Loss-kernel non-finite cases (LDL `var**0.2`, MS-SSIM negative powers, NCC divide-by-zero) now abort runs instead of silently continuing → fix them in the losses topic first.
3. DDP/compile unproven by the probe: `requires_grad` flips on a compiled D → two cached graphs; DDP broadcasts buffers on every wrapped forward (two D forwards before one backward) — plain for SN discriminators, a version-counter risk for BN ones (Codex #6). Step E before claiming support.
4. Log semantics change (window mean, undivided D terms, `l_d_total` = sum): tensorboard curves shift by constants.
5. Accumulation equals full-batch only for additive objectives; MS-SSIM's batch-level statistic (ssim_loss.py:151) and D's stochasticity are not batch-linear (inherent, as today).
6. `Ema.src` / `Phase.params` / `d_params` are cached parameter lists: any later module replacement (QAT prepare, fusion) must rebuild them; today's QAT path (train.py:414-437) is dead and runs after optimizer construction.

## 6. What I would NOT do
- No label/`isinstance` dispatch, loss wrapper hierarchy, dynamic loss scheduler, `is_active` gating, automation thresholds (traiNNer `sr_model.py:452-700`); scheduling lives in config (`wavelet_init`) or in a term's own `evaluate`.
- Not converting kernels to `forward(ctx)`; not merging G and D scalers; not re-adding `net_d_iters`, the GAN weight schedule, SAM, D-EMA, `warmup_iter`.
- Not making bf16 strict by default (owner's no-per-iteration-sync requirement; the knob exists); not adding per-tensor `torch.where` EMA shielding (~3 launches per parameter per update for a rare event, and it does not repair live weights or optimizer state).
- Not compiling the whole trainer, not adding a callback framework, not shipping `[[train.losses]]` before the legacy builder is verified.
- Not touching OTF degradations, augmentations, dataloader, checkpoint file format, validation tiling; only their interfaces to the step (`feed_data`, `eval_weights`, `check_finite`, "save after `stepped`") are fixed here.

## 7. Open questions
1. bf16 default: non-strict (proposed, no per-update sync) or strict (Codex's preference for reference quality)? Middle ground: `strict_finite = true` for the first N updates of a fresh run.
2. Validation/export weights under schedule-free: EMA(y) (proposed, today's behaviour) or live x? Expose both?
3. Keep ECO (proposed) or delete (zero configs)?
4. Checkpoint topic: `params`/`params_ema` keys, scalers and counters in state, versioned migration boundary vs exact resume of historical runs.
5. Which DDP settings and BN discriminators (patchgan) must be supported initially?

## 8. Codex review — adopted / rejected
Adopted: #1 (norm overflow → elementwise `found_inf`; measure-only norm when clipping is off), #2 (strict never bypasses the scaler), #5 (freeze mask; metagan dropout caveat), #7 (Parameter objects in `Ema.src`; restore prior modes; no SF swap when evaluating EMA), #8 (QAT ordering note), #9 (`evaluate`/`compute` split; fp32 derived inputs; cached adv view), #10 (single transfer; hidden syncs listed per loss), #11 (reduce attempted count in the same units; collective on all ranks), #12 (clock contract: bool = attempted window; `it` = completed+1; scheduler per attempted update; probe's clipping comment removed — parity holds with clipping on), #13 (clip default 1.0; EMA stays eval; builder validation; defer generic list; check before save/validation/shutdown).
Rejected: strict skipping as the bf16 default (contradicts the owner's explicit no-sync requirement; offered as a knob and as open question 1); per-tensor `torch.where` EMA shield (cost, and it leaves live weights/optimizer state corrupted anyway).
Deferred to other topics: #3 (checkpoint contents, SF snapshot pairing), #4 BN buffer policy, #6 DDP/compile probe (step E).
