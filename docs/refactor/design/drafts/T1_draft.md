# T1 — Loss pipeline and training step (synthesis draft)

Inputs merged: the Opus position, the Fable proposal (design `T1_loss_pipeline_and_training_step.md`, runnable `skeleton/`, 36-check GPU probe, Codex review), and the Codex proposal. Conflicts were settled against the code in `/home/korvin/apps/neosr-dev` (read-only), the 135 `options/*.toml` files, and one new GPU probe (`design/probe_inputs.py`).

## 0. Facts that decide the conflicts

| Fact | Evidence |
|---|---|
| All 85 training configs use `use_amp=true` and `bfloat16=true`. None uses fp16. The 44 other configs are test configs. | grep of `options/*.toml` |
| Every optimizer table (170 of 170) is `adan_sf`. No config uses AdamW, fused or otherwise. | same |
| 85 configs set `mssim/ldl/gan/consistency/fdl` (84 use fdl), with a `metagan` D. 82 set `wavelet_guided=true` with `wavelet_init=80000`. 1 sets `match_lq_colors`. 2 set `grad_clip=false`. 7 set `clamp=false`. | same |
| `accumulate`, `sam`, `eco` and schedulers appear only in comments. No config enables them. | `options/train_esrgan.toml:17,54-57` |
| Loss weight defaults are 1.0 everywhere except `gan_loss` (0.1). 15 kernels multiply by the weight internally. `ResNetPerceptualLoss` uses `perceptual_weight` instead. | `basic_loss.py:53,85,127,219`, `ssim_loss.py:126`, `gan_loss.py:26,84`, `resnet_perceptual_loss.py:37,91`, … |
| The G-side GAN term runs D under `inference_mode` and in eval mode, so it is dead. D trains on wavelet HF bands while G's (dead) query uses RGB. D logs `(real+fake)/2` but backpropagates the sum. | `image.py:607-616`, `:651-700` |
| The wavelet view runs inside bf16 autocast. Its Y input spans [16, 235], so bf16 (≈8 mantissa bits, absolute step ≈1 at 235) quantizes away the HF bands it is meant to expose. | `image.py:538-541` within the autocast at `:506`; `wavelet_guided.py:386-432` (`F.conv2d`, :159) |
| The micro counter is incremented twice per micro-batch. The EMA advances even when the scaler skipped. NaN is checked with a host sync on every micro-batch. | `image.py:502,735`, `:765-769`, `:717` |
| Stock `GradScaler.step` syncs with `.item()` to decide whether to step. | `torch/amp/grad_scaler.py:356` |
| `loss.backward(inputs=g_params)` through a train-mode D gives the same G grads as the freeze-toggle approach. D `.grad` stays `None`, an unused G param is tolerated, and the cost is the same (3.51 vs 3.51 ms over 100-iteration runs). | new probe `design/probe_inputs.py`, RTX 4090, torch 2.9.1 |

## 1. Conflict resolution

| Question | Opus | Fable | Codex | **Decision**: why |
|---|---|---|---|---|
| Loss interface | numeric `forward(sr, target)` + mixin `compute(ctx)` | same, plus `evaluate(ctx)` override point, `inputs` tuple | `forward(ctx) -> LossResult`, collection applies weights | **Fable/Opus.** `forward(*tensors)` stays a pure kernel, so it is unit-testable and the 15-file diff is "delete the weight multiply". `evaluate(ctx)` is the only override; `compute` cannot be bypassed. `LossResult` multi-term returns are unnecessary: the one multi-part loss (GAN D) logs its parts through `ctx.log`. |
| Input selection | per-loss `target` | `inputs=("sr","gt")` tuple | `target` + `PairLoss.select_target` | **`target` attribute** (`"gt"` or `"lq"`→`ctx.lq_up`). Every pair kernel takes exactly (sr, reference); anything else overrides `evaluate`. A generic `inputs` tuple has no second user. |
| Where the weight lives | base | base `compute` (ctor arg) | collection | **Base `compute`, value set by the builder.** The builder pops `loss_weight` (class default `Loss.loss_weight=1.0`, `gan_loss.loss_weight=0.1`), so kernels never see it (Codex #4) and it is applied in one place (Fable). |
| GAN D input | — | `ctx.cache["adv_sr"]` written by the G term | `replace(ctx, sr=sr.detach())`, view recomputed | **Recompute** `adv(sr.detach())` in the D term. No mutable cache and no G→D ordering coupling, and it still works when the G weight is 0 and the G term is pruned. The cost is one SWT on a detached tensor (≈0.2 ms, Fable probe). |
| Wavelet view precision | — | not addressed | not addressed | **fp32, outside autocast** (new finding, row 6 of §0). One module built once with filters as frozen parameters and no `@no_grad`, applied identically to sr (with grad) and gt (Fable, Codex). |
| D frozen during the G pass | `requires_grad_(False)` | freeze mask over the initially trainable params | freeze/restore in `finally` | **Neither: `backward(inputs=phase.params)`.** Probe: identical G grads, D grads untouched, same speed. No flag toggling, no mask, nothing to restore on exception, and QAT/partial freezes are irrelevant. The G pass calls the **bare** D (DDP-unwrapped, compile kept), so D's DDP reducer never sees a backward without D grads. |
| D mode | train | train | train | **train, always** (agreed). D also sees the G-pass forward, which advances SN and BN buffers as in BasicSR. MetaGAN dropout 0.2 is now active in the G pass (Risk 1). |
| fp16 scaler | 1 shared, fp16 only | per phase, fp16 only | per phase, fp16 only | **Per phase, fp16 only, `init_scale=2**5` for both.** Today D uses the default 2**16 (`image.py:153`). |
| Non-finite detection | at print_freq | private `_amp_foreach_non_finite_check_and_unscale_`, on device | fused-AdamW adapter with private `_found_inf_per_device` plus `opt.found_inf` | **Fable.** The Codex adapter covers only fused Adam(W), which 0 of 170 optimizer tables use. Under fp16 the stock scaler's one sync per *update* (not per micro-batch) is inherent and accepted; no config uses fp16. bf16/fp32 default: no sync, abort at the next check. |
| "Stuck" detection | — | `skipped == attempted` in the window | consecutive-failure threshold | **Codex: on-device consecutive streak.** Fable's window test can abort on one legitimate fp16 skip when `check_finite()` runs on a 1–2-update window before a save. The streak counter has no such false positive. |
| Counter units | micro % accum | `it` = completed updates + 1 | keep raw micro `current_iter` for `wavelet_init` | **Fable.** Identical at `accumulate=1`, which is every config. Updates are the meaningful unit, and train.py's re-divisions disappear (`train.py:454,467,536,565`). |
| Scheduler clock | real optimizer steps | attempted updates | successful updates, tensor LRs | **Attempted updates.** No config uses a scheduler. Success-gated LRs need a sync or device-side schedule code for no user. |
| LQ fed to G vs LQ for colour targets | — | ECO overwrites `lq` | `g_input` separate from `lq` | **Codex.** `Context(lq=original, g_in=generator input)`. `lq_up` and `sr_ema` use the right one under ECO. |
| ECO | — | keep as pure 20-line transform | keep, preliminary pass in eval mode | **Keep Fable's pure transform, today's train-mode preliminary pass.** Zero configs use it, so an unrequested behaviour change is not worth a compile recompile on mode toggle. It is deletable at no cost if the owner prefers. |
| SAM (fsam) | keep only if cheap | delete | delete | **Delete.** Reject `sam` with a "removed" message. |
| Zero-weight terms | — | — | prune G terms with weight 0, keep D term | **Adopt** (one filter in the builder). |
| Scalar output | — | — | require 0-dim | **Adopt** as a shape check in `compute`. It needs no sync. |
| Generic list | every `*_opt` generic | legacy table now, `[[train.losses]]` later | both forms now | **Both now, one loop.** Every `train.<slot>_opt` is read generically (log key `l_g_<slot>`, alias `pixel→pix`, `perceptual→percep`). `[[train.losses]]` entries (`name`, `type`, …) feed the same loop in ~1 extra line. Duplicate keys raise. |
| `l_d_total` | — | sum | sum | **Sum** (what is optimized). `l_d_real`/`l_d_fake` stay logged. |
| D-EMA | — | delete | — | **Delete.** Its only effect is that resume continues D from its EMA (`image.py:999`). |
| Schedule-free | — | `eval_weights()` context, EMA of y | representation adapters, EMA of x | **Fable**, with x-vs-y for EMA left as an open question. Today's checkpoints are EMA(y). Codex's adan_sf warmup defect (`adan_sf.py:260,315`) goes to the optimizer topic. |

## 2. Core skeleton — `neosr/losses/base.py` + builder in `neosr/losses/__init__.py` + GAN (104 lines)

```python
TARGETS = {"gt": "gt", "lq": "lq_up"}
SLOT_KEY = {"pixel": "pix", "perceptual": "percep"}           # legacy log names; others: l_g_<slot>

class Context:
    """What one generator forward exposes to the losses. Built by the trainer; losses only read it."""
    def __init__(self, *, sr, gt, lq, g_in, it, net_d=None, net_g_ema=None):
        self.sr, self.gt, self.lq, self.g_in, self.it = sr, gt, lq, g_in, it
        self.net_d, self.net_g_ema = net_d, net_g_ema
        self.logs: dict[str, Tensor] = {}                       # detached device scalars

    @cached_property
    def lq_up(self) -> Tensor:                                  # colour target (was match_lq_colors)
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            up = F.interpolate(self.lq.float(), size=self.sr.shape[-2:], mode="bicubic")
        return up.clamp_(1 / 255, 1)

    @cached_property
    def sr_ema(self) -> Tensor:                                 # opt-in teacher (e.g. EMA-refined LDL)
        with torch.no_grad():
            return self.net_g_ema(self.g_in)

    def log(self, key: str, value: Tensor) -> None:
        self.logs[key] = value.detach()

class Loss(nn.Module):
    """Objective term. forward(*tensors) is the unweighted kernel; trainers call compute(ctx) only."""
    loss_weight = 1.0; target = "gt"; phase = "g"; log_key = ""; fp32 = False

    def expand(self) -> list["Loss"]:
        return [self]

    def evaluate(self, ctx: Context) -> Tensor:                 # override for models/extra inputs
        sr, ref = ctx.sr, getattr(ctx, self.target)
        return self(sr.float(), ref.float()) if self.fp32 else self(sr, ref)

    def compute(self, ctx: Context) -> Tensor:
        with torch.autocast("cuda", enabled=False) if self.fp32 else nullcontext():
            value = self.evaluate(ctx) * self.loss_weight       # the only place a weight is applied
        if value.ndim:
            raise ValueError(f"{type(self).__name__} must return a scalar")
        if self.log_key:
            ctx.log(self.log_key, value)
        return value

def build_losses(train: dict, device) -> tuple[list[Loss], list[Loss]]:
    if removed := {"gw_opt", "color_opt", "luma_opt"} & train.keys():
        raise ValueError(f"removed loss options: {sorted(removed)}")
    specs = [(k[:-4], dict(v)) for k, v in train.items() if k.endswith("_opt") and v]
    specs += [(s.pop("name"), s) for s in map(dict, train.get("losses", ()))]
    terms: list[Loss] = []
    for name, s in specs:                                       # legacy globals -> per-loss keys
        if name in ("consistency", "msswd") and train.get("match_lq_colors"):
            s.setdefault("target", "lq")
        if name == "gan" and train.get("wavelet_guided"):
            s.setdefault("wavelet_init", train.get("wavelet_init", 0))
        cls = LOSS_REGISTRY.get(s.pop("type"))
        weight = s.pop("loss_weight", s.pop("perceptual_weight", cls.loss_weight))
        target = TARGETS[s.pop("target", "gt")]
        loss = cls(**s)
        loss.loss_weight, loss.target = float(weight), target
        loss.log_key = loss.log_key or f"l_g_{SLOT_KEY.get(name, name)}"
        terms += [t for t in loss.expand() if t.phase == "d" or t.loss_weight]
    keys = [t.log_key for t in terms if t.log_key]
    if not any(t.phase == "g" for t in terms) or len(set(keys)) != len(keys):
        raise ValueError(f"need >=1 generator loss and unique keys, got {keys}")
    nn.ModuleList(terms).to(device, memory_format=torch.channels_last).requires_grad_(False).eval()
    return [t for t in terms if t.phase == "g"], [t for t in terms if t.phase == "d"]

class gan_loss(Loss):                                           # [train.gan_opt] -> G term + D term
    loss_weight = 0.1; log_key = "l_g_gan"
    def __init__(self, gan_type="bce", real_label_val=1.0, fake_label_val=0.0, wavelet_init=None):
        super().__init__()
        self.crit = {"bce": nn.BCEWithLogitsLoss, "mse": nn.MSELoss, "huber": nn.HuberLoss}[gan_type]()
        self.labels = (fake_label_val, real_label_val)
        self.view = WaveletHF() if wavelet_init is not None else None   # built once; filters frozen
        self.view_from = wavelet_init or 0

    def adv(self, ctx, x):                                      # same view for G and D
        if self.view is None or ctx.it < self.view_from:
            return x
        with torch.autocast("cuda", enabled=False):             # bf16 erases HF bands of Y in [16,235]
            return self.view(x.float())

    def term(self, logits, real: bool):
        return self.crit(logits, torch.full_like(logits, self.labels[real]))

    def evaluate(self, ctx):                                    # bare D, train mode; grads reach sr only
        return self.term(ctx.net_d(self.adv(ctx, ctx.sr)), True)

    def expand(self):
        return [self, gan_d_loss(self)]

class gan_d_loss(Loss):                                         # unweighted (gan_loss.py:83-84)
    phase = "d"
    def __init__(self, g: gan_loss):
        super().__init__(); self.g = g
    def evaluate(self, ctx):
        real = ctx.net_d(self.g.adv(ctx, ctx.gt))
        fake = ctx.net_d(self.g.adv(ctx, ctx.sr.detach()))
        l_real, l_fake = self.g.term(real, True), self.g.term(fake, False)
        for k, v in (("l_d_real", l_real), ("l_d_fake", l_fake),
                     ("out_d_real", real.mean()), ("out_d_fake", fake.mean())):
            ctx.log(k, v)
        return l_real + l_fake
```

Notes:
- The trainer loop is `sum(t.compute(ctx) for t in terms)`. It has no type checks, no if-chain, and no knowledge of GAN. A loss that needs a model reads it from `ctx` (`net_d`, `net_g_ema`). A loss that needs other data overrides `evaluate`. The builder branches only on **legacy slot names** to translate old global keys. The loss loop never branches.
- Freezing: `requires_grad_(False).eval()` runs once over every term **after** construction. That covers DISTS `alpha/beta` (created after today's freeze, `dists_loss.py:92-100`), the VGG/ResNet/DINO backbones, and the SWT filters. D and the EMA generator are context references, not children, so freezing losses can never freeze a trained network. `gan_d_loss.g` is a child, so the criterion and view appear twice in the ModuleList. Parameters are deduplicated, so this is harmless.
- `WaveletHF` is `wavelet_guided.py:386-432` as an `nn.Module`: sym19 filters and band scales unchanged, one `SWTForward` built in `__init__`, `@torch.no_grad` removed, input RGB→Y, output 3 HF channels.

## 3. Training step skeleton — `neosr/models/phase.py` + the `image` step (105 lines)

```python
class Phase:
    """One network's objective: its terms, optimizer, fp16 scaler, clipping, non-finite accounting."""
    def __init__(self, opt, terms, *, log_key, amp_dtype, use_amp, accum, grad_clip=1.0,
                 strict=False, schedulers=()):
        self.opt, self.terms, self.log_key = opt, terms, log_key
        self.params = [p for g in opt.param_groups for p in g["params"] if p.requires_grad]
        self.autocast = {"device_type": "cuda", "dtype": amp_dtype, "enabled": use_amp}
        fp16 = use_amp and amp_dtype is torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=fp16, init_scale=2.0**5)
        self.strict = strict and not fp16                       # never bypass the scaler's update()
        self.protected = fp16 or self.strict                    # a bad grad can never reach the weights
        self.inv_accum, self.grad_clip, self.schedulers = 1.0 / accum, grad_clip, list(schedulers)
        z = lambda dt=torch.long: torch.zeros((), dtype=dt, device=self.params[0].device)
        self.one, self.bad, self.streak, self.grad_norm = z(torch.float) + 1, z(), z(), z(torch.float)

    def backward(self, ctx) -> None:
        with torch.autocast(**self.autocast):
            total = sum(t.compute(ctx) for t in self.terms)
        ctx.log(self.log_key, total)                            # undivided micro-batch objective
        self.scaler.scale(total * self.inv_accum).backward(inputs=self.params)

    def step(self) -> Tensor:                                   # device bool, no host sync by default
        self.scaler.unscale_(self.opt)                          # no-op unless fp16
        grads = [p.grad for p in self.params if p.grad is not None]
        found = torch.zeros((), device=self.one.device)
        torch._amp_foreach_non_finite_check_and_unscale_(grads, found, self.one)  # GradScaler's op
        # DDP: grads are already all-reduced, so every rank reaches the same verdict.
        finite = found == 0
        norm = (clip_grad_norm_(self.params, self.grad_clip, foreach=True) if self.grad_clip
                else torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads))))
        if not (self.strict and not finite.item()):             # strict: 1 sync/update, clean skip
            self.scaler.step(self.opt)                          # fp16: scaler skips inf itself
            self.scaler.update()
        self.opt.zero_grad(set_to_none=True)
        for s in self.schedulers:                               # clock = attempted updates
            s.step()
        self.bad += ~finite; self.streak.mul_(~finite).add_(~finite); self.grad_norm.copy_(norm)
        return finite

class Ema:                                                      # of the bare, uncompiled G
    def __init__(self, bare: nn.Module, decay: float):
        self.module = copy.deepcopy(bare).eval().requires_grad_(False)
        self.w = 1.0 - decay
        self.dst, self.src = list(self.module.parameters()), list(bare.parameters())
        self.dst_b, self.src_b = list(self.module.buffers()), list(bare.buffers())

    @torch.no_grad()
    def update(self, finite: Tensor) -> None:                   # skipped update -> weight 0
        torch._foreach_lerp_(self.dst, self.src, self.w * finite)
        if self.dst_b:
            torch._foreach_copy_(self.dst_b, self.src_b)

# --- image model (otf inherits it unchanged) ---------------------------------------------
def optimize_parameters(self, it: int) -> bool:
    """One micro-batch. it = completed updates + 1. Returns True when an update was attempted."""
    g_in, gt = self.eco(self.net_g, self.lq, self.gt, it) if self.eco else (self.lq, self.gt)
    with torch.autocast(**self.g.autocast):
        sr = self.net_g(g_in)
        if self.clamp:
            sr = sr.clamp(1 / 255, 1)
    ctx = Context(sr=sr, gt=gt, lq=self.lq, g_in=g_in, it=it, net_d=self.bare_d,
                  net_g_ema=self.ema.module if self.ema else None)
    boundary = (self.micro + 1) % self.accum == 0
    with no_sync(self.net_g, boundary):
        self.g.backward(ctx)
    if self.d is not None:
        ctx.net_d = self.net_d                                  # DDP-wrapped D for its own pass
        with no_sync(self.net_d, boundary):
            self.d.backward(ctx)
    self.micro += 1
    if boundary:
        finite = self.g.step()
        if self.d is not None:
            self.d.step()
        if self.ema is not None:
            self.ema.update(finite)
    for k, v in ctx.logs.items():                               # on-device window sums
        s = self.log_sum.get(k)
        self.log_sum[k] = v.float().clone() if s is None else s.add_(v)
    self.log_n += 1
    return boundary

def get_current_log(self, reset: bool = True) -> dict[str, float]:
    """One stack + one all_reduce + one host copy; raises before anything is saved from bad weights."""
    phases = {n: p for n, p in (("g", self.g), ("d", self.d)) if p is not None}
    keys = list(self.log_sum)
    vec = torch.stack([self.log_sum[k] / max(self.log_n, 1) for k in keys]
                      + [t for p in phases.values() for t in (p.grad_norm, p.bad.float(), p.streak.float())])
    if dist.is_initialized():
        dist.all_reduce(vec, op=dist.ReduceOp.AVG)              # counters are rank-identical
    vals = vec.tolist()
    out, i = dict(zip(keys, vals)), len(keys)
    for n, p in phases.items():
        norm, bad, streak = vals[i:i + 3]; i += 3
        if (bad and not p.protected) or streak >= self.max_skip_streak:
            raise RuntimeError(f"non-finite gradients in {n}: {int(bad)} this window, streak "
                               f"{int(streak)}; restart from the last checkpoint (lower lr / strict_finite)")
        out[f"grad_norm_{n}"], out[f"skipped_{n}"] = norm, bad
    if reset:
        self.log_sum.clear(); self.log_n = 0
        for p in phases.values():
            p.bad.zero_()
    return out

check_finite = functools.partialmethod(get_current_log, reset=False)   # before save/validation/exit
```

`no_sync(net, boundary)` is `net.no_sync() if not boundary and hasattr(net, "no_sync") else nullcontext()`. `self.bare_d = get_bare_model(net_d)` unwraps DDP and keeps compile. `max_skip_streak` defaults to 100 (`train.max_skip_streak`).

train.py contract:

```python
if not model.optimize_parameters(update + 1):
    continue
update += 1
```

After that, `print_freq`, `save_checkpoint_freq` and `val_freq` count in `update`. Logging calls `get_current_log()`. Save, validation and the interrupt handler call `check_finite()` first. The partial window carries across epochs. On resume `micro=0`, so a partial window is dropped. State `iter` = updates. `eval_weights(swap_sf=self.ema is None)` (Fable §10) wraps validation and export: SF `opt.eval()`/`train()` and module modes, restored in `finally`.

### Step semantics, one line each
- **Accumulation:** one counter (`micro`), owned by the model. `boundary` is computed before the backward so DDP `no_sync` applies to non-boundary micro-batches. The divisor is applied once (`inv_accum`).
- **AMP:** bf16/fp32 have no scaler. fp16 uses one scaler per phase. Clipping runs after unscale, once per window. `grad_clip=true`→1.0, `false`→0 means measure only (no `clip(inf)` `inf/inf` NaN, Fable probe 11). SOAP-requires-no-clip validation stays.
- **G/D:** G pass: bare D in train mode, `backward(inputs=G params)`, so the adversarial gradient reaches G and none reaches D. The G backward finishes before the D forwards (in-place SN/BN buffer updates are safe). D pass: fresh forwards on `gt` and `sr.detach()`. Logits are never shared between phases.
- **EMA:** advances only on boundaries, gated on device by `finite`. The gate protects against fp16/strict skips. In unprotected bf16, `lerp` with weight 0 still propagates NaN (Fable-verified), which is why such a run aborts at the next check and never writes a checkpoint.
- **NaN/inf:** no per-micro-batch sync. fp16: scaler skip, logged `skipped_*`, fatal only after `max_skip_streak` consecutive skips. bf16/fp32 default: abort at the next `get_current_log`/`check_finite`, so at most `print_freq` updates are lost and no corrupted checkpoint is written. `train.strict_finite=true`: one `.item()` per update and a clean skip.
- **Logging:** fixed keys with detached device sums. Values are window means of weighted terms per micro-batch. Keys stay the same (`l_g_*`, `l_g_total`, `l_d_real`, `l_d_fake`, `l_d_total`, `out_d_*`), with `grad_norm_g/d` and `skipped_g/d` added. `l_d_total` = real + fake. `reduce_loss_dict` and its `.item()` loop (`base.py:551-559`) are deleted.
- **SAM:** deleted (`optimizers/fsam.py`, `image.py:301-310,383-413`). The config key `sam` raises "removed". Reason: it owns a second optimizer that is never checkpointed, replays the stateful closure that also trains D, and is incompatible with accumulation and AMP. 0 configs use it.
- **ECO:** `Eco.__call__(net_g, lq, gt, it) -> (g_in, gt)` as a pure function (Fable `skeleton/step.py:18-37`). It no longer mutates `self.gt`, and `ctx.lq` keeps the original.
- **Schedule-free:** kept. EMA keeps averaging the training representation y, as today. `update_learning_rate`, `warmup_iter`, `_get_init_lr` (crash at `base.py:253`) and `get_current_gan_weight` are deleted; no config uses them.

## 4. Mapping of every current loss

Every kernel drops its internal `* loss_weight` and its `loss_weight` ctor argument. The builder owns the weight. "fp32" means `fp32 = True` (the kernel runs outside autocast).

| Class (file:forward) | slot → log key | inputs | change beyond "drop weight" |
|---|---|---|---|
| `L1Loss`, `MSELoss`, `HuberLoss`, `chc_loss` (`basic_loss.py:45,77,114,180`) | `pixel_opt` → `l_g_pix` | sr, target | `reduction="none"` is rejected by the scalar check |
| `mssim_loss` (`ssim_loss.py:112`) | `mssim_opt` → `l_g_mssim` | sr, gt; fp32 | positive floor before fractional powers (losses topic) |
| `ncc_loss` (`ncc_loss.py:36`) | `ncc_opt` → `l_g_ncc` | sr, gt; fp32 | per-(N,C) flatten, eps in the denominator (losses topic) |
| `kl_loss` (`kl_loss.py:21`) | `kl_opt` → `l_g_kl` | sr, gt | keep the internal 0.03 |
| `fdl_loss` (`fdl_loss.py:318`) | `fdl_opt` → `l_g_fdl` | sr, gt | backbone frozen/eval by the builder |
| `consistency_loss` (`consistency_loss.py:146`) | `consistency_opt` → `l_g_consistency` | sr, gt\|lq_up; fp32 | `target` key; legacy `match_lq_colors` → `target="lq"`; `if cosim < 1e-3` (:189) → tensor select (hidden sync) |
| `msswd_loss` (`msswd_loss.py:166`) | `msswd_opt` → `l_g_msswd` | sr, gt\|lq_up | same `target` rule; accumulate inside the scale loop (losses topic) |
| `vgg_perceptual_loss` (`vgg_perceptual_loss.py:185`) | `perceptual_opt` → `l_g_percep` | sr, gt | frozen/eval backbone |
| `ResNetPerceptualLoss` (`resnet_perceptual_loss.py:59`) | `perceptual_opt` → `l_g_percep` | sr, gt | `perceptual_weight` read by the builder as an alias of `loss_weight` (0 configs); undefined return when weight 0 disappears |
| `dists_loss` (`dists_loss.py:144`) | `dists_opt` → `l_g_dists` | sr, gt | alpha/beta frozen by the builder (today they train) |
| `ldl_loss` (`ldl_loss.py:98`) | `ldl_opt` → `l_g_ldl` | sr, gt; fp32 | stop storing `self.output/self.gt` (:100-101); detached artifact map; an EMA-refined variant would override `evaluate` with `ctx.sr_ema` and be opt-in |
| `ff_loss` (`ff_loss.py:118`) | `ff_opt` → `l_g_ff` | sr, gt; fp32 FFT | `.item()` asserts (:102) removed |
| `gan_loss` (`gan_loss.py:64`) | `gan_opt` → `l_g_gan` + D term (`l_d_real`, `l_d_fake`, `out_d_*`; phase total `l_d_total`) | ctx.net_d, sr, gt | rewritten (§2); `update_loss_weight` deleted |
| `wavelet_guided` (`wavelet_guided.py:386`) | globals `wavelet_guided`/`wavelet_init` → `gan_opt.wavelet_init` | — | becomes the `WaveletHF` view: fp32, differentiable, built once |
| `loss_util.get_refined_artifact_map`; `default.py` `gw/color/luma_opt` | — | — | deleted; the builder raises "removed" |

Generic list: `[[train.losses]] name="edge" type="L1Loss" loss_weight=0.5 target="lq"` gives the log key `l_g_edge`.

## 5. Migration (same letter = disjoint files, parallel)
- **A1** `neosr/losses/base.py` + `losses/__init__.py` (builder). Verify: dry-run of all 135 configs against a recording registry. Term set and keys must equal today's `if self.cri_*` set (`image.py:179-273`). `options/train_dctlsa.toml:88` already fails TOML parsing (pre-existing).
- **A2** `gan_loss.py` + `wavelet_guided.py` (`WaveletHF`). Verify: G-only GAN gradient non-zero; D grads `None` after the G pass; view switches at `wavelet_init` for both terms; `gradcheck` through the view; fp32 vs bf16 HF-band error (expect large under bf16).
- **A3** `neosr/models/phase.py` (Phase, Ema), `neosr/models/eco.py`. Verify: Fable's `skeleton/probe.py` ported to this variant (`backward(inputs=)`, streak), in fp32/bf16/fp16.
- **B1** (after A1), one worker per `*_loss.py`. Verify: `allclose(new(x, y) * w, old(x, y))`; `grep -n loss_weight neosr/losses` returns only `base.py` and `gan_loss`'s class attribute.
- **B2** (after A1-A3) `models/image.py`: replace `:88-351` with phase construction (~50 lines) and `:462-770` with §3. `models/base.py`: delete `reduce_loss_dict`, `update_learning_rate`, `_get_init_lr`, `_set_lr`, `update_loss_weights`, `get_current_gan_weight`, D-EMA; add `eval_weights` and `check_finite` to validation and save. Verify: 300 updates of `train_esrgan2.toml`; key set = today's + `grad_norm_*`/`skipped_*`; `out_d_*` bounded.
- **B3** (after A3) `train.py`, `utils/logger.py`: loop contract, removal of the accumulate re-divisions (`train.py:448-568`), `check_finite` on interrupt. Verify: `accumulate=1` vs `2` with half the batch gives the same update count and cadence.
- **C** (after B2): delete `models/default.py`, `simple_otf.py`, `eqr_otf.py`, `pancam_otf.py`, `image.py.helaman`, `optimizers/fsam.py`, `losses/loss_util.py`. Verify: imports and the 135-config dry-run.
- **D** (after B2): `tests/test_step.py` = the probe as pytest. **E** (optional): two-rank DDP + `compile=true` smoke. This confirms DDP hooks fire under `backward(inputs=)` on the wrapped G, and that `no_sync` holds. Fallback if not: the freeze-toggle G pass from Fable.

## 6. Risks
1. **Intended dynamics change.** G now trains adversarially, through a train-mode D (MetaGAN dropout 0.2 and DropPath active, `metagan_arch.py:40-43,103`), and on HF bands from update 80000. `gan_opt.loss_weight=0.3` was tuned against a dead term, so retuning may be needed. It is not automated.
2. **bf16 unprotected by default.** A non-finite update corrupts live and EMA weights until the next check (≤ `print_freq` updates). It never reaches a checkpoint. Loss kernels with non-finite edge cases (MS-SSIM powers, NCC denominators, LDL `var**0.2`) will now abort, so fix them in the losses topic first.
3. `backward(inputs=)` under DDP and `compile` is unverified on 2 ranks (step E). The private op `torch._amp_foreach_non_finite_check_and_unscale_` is GradScaler's own and stable since 1.6. It is confined to `Phase.step`.
4. Cached parameter lists (`Phase.params`, `Ema.src`) assume modules are not replaced after construction. QAT prepare currently runs after optimizer construction (`train.py:414-437`) and must move before it.
5. Accumulation ≠ full batch for batch-statistic losses (MS-SSIM) and train-mode D buffers. This is the same as today.

## 7. Would not do
`isinstance`/label dispatch in the loop; `forward(ctx)` kernels; `LossResult`/collection objects; a shared mutable ctx cache; a loss dependency graph; the fused-AdamW `found_inf` adapter (0 users, private scaler state); success-gated tensor LR schedules; per-tensor `torch.where` EMA shielding; strict bf16 by default; re-adding SAM, D-EMA, `warmup_iter`, `net_d_iters` or a GAN weight schedule; compiling the trainer; a callback framework.

## 8. Open questions for the owner
1. Default for bf16: non-strict (proposed, no sync) or `strict_finite=true` (one `.item()` per update and clean skips)?
2. Validation and export under schedule-free: EMA(y) (proposed, as today) or live x, or both?
3. ECO: keep it (pure, ~20 lines, 0 configs) or delete it?
4. Is retuning of `gan_opt.loss_weight` acceptable now that the adversarial term is live, or should a first release ship `wavelet_init`/GAN behaviour behind a flag?
5. Checkpoint layout (scalers, `micro`/update counters, SF pairing) is deferred to the checkpoint topic. This design only fixes the interfaces: save after `boundary`, `check_finite`, `eval_weights`, `Ema.module.state_dict()`.
