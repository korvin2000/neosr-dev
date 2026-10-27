# Stage 2 audit — lane: optimizers and their trainer integration

Scope: `neosr/optimizers/{adan_sf,adamw_sf,adan,__init__}.py`, `neosr/models/phase.py` (Phase,
is_schedule_free), `neosr/models/image.py` (eval_weights, export_state_dicts), `neosr/models/base.py`
(OPTIMIZERS), `tests/test_optimizers.py`, `tests/test_phase*.py`, the 168 `adan_sf` template sections,
competitor `traiNNer/optimizers/`. Codex items: F01 (in depth), F19 (optimizer part).

All probes ran on CPU (`CUDA_VISIBLE_DEVICES=`), torch 2.9.1. Scripts and raw outputs:
`/tmp/claude-1000/-home-korvin-apps-neosr-dev/c1b40b02-f454-4498-9d89-be0afa4e2973/scratchpad/optimizers/`
(`sf_variants.py`, `probe_f01.py`, `probe_toy.py`, `probe_betas.py`, `probe_ops.py`, `probe_params.py`,
`probe_state.py`, `out_*.txt`). Nothing in the repository was changed except this file.

## 0. Summary

1. **F01 confirmed and extended (P1).** The schedule-free branch of `adan_sf` is not a schedule-free
   optimizer: `z` follows the *raw, unpreconditioned* gradient, the live weights `y` are pulled toward
   that `z` every step, the first step does not move `y` at all, the evaluation weights move uphill, and
   step-1 weight decay is erased. New evidence beyond Codex: on an ill-conditioned quadratic (cond 1e3,
   lr 1e-2) the current optimizer **diverges to 5e29** while every other optimizer in the comparison
   converges, and on a tiny conv SR problem it reaches a 2.2x higher loss at step 100 than the corrected
   recurrence (0.596 vs 0.270). In real SR runs the gradient clip to 1.0 keeps the raw `z` step tiny, so
   `z` stays close to the initial weights and the optimizer behaves like "Adan with a continuous pull back
   toward the initialisation"; it learns, but slower than it should, and the D is handicapped the same way.
2. **The G4.M2 consensus formula must not be implemented as written (P1, decision).** Keeping Adan's
   first-moment EMA `m` (beta1 = 0.98) *and* using beta1 as the schedule-free interpolation is double
   momentum: on the conv SR toy it ends at 0.48 (vs 0.057 for the sound variant and 0.069 for the current
   code) and at lr 5e-3 it does not learn at all (0.66). The sound schedule-free Adan replaces `m` by the
   SF interpolation, exactly as SF-AdamW drops Adam's `m`.
3. **A mathematically sound schedule-free Adan** is derived in §2, with a foreach implementation sketch
   (§3) that needs 3 (or 4) state buffers per parameter instead of 5 and 13-18 foreach ops per step instead
   of 20. With `beta2 = 0` it reduces *exactly* to the reference SF-AdamW with `(beta1, beta3)` (verified
   numerically to the last digit), which gives a built-in control arm for the A/B.
4. **The Nesterov gradient-difference term is doubtful for SR (P2, tuning).** On the mini-batch MLP toy it
   hurts (final 0.26 vs 0.034 without it); on the conv SR toy it is neutral to slightly negative. It must
   be an arm of the A/B, not an assumption.
5. `adamw_sf` matches the facebookresearch reference line by line (max |x - x_ref| 1.4e-6 after 300 noisy
   steps, both foreach and loop paths); `adan` matches the official sail-sg kernel (foreach == single
   tensor == `adan_sf(schedule_free=False)` bit-exactly for `no_prox=True`). No deviations to fix there.
6. Speed/memory are secondary: the optimizer is a few ms of a 300-500 ms SR update. The real wins are the
   state buffers (ATD 20.3M params: 386 MiB -> 232 MiB (3 buffers) or 155 MiB (2 buffers), plus 46-69 MiB
   on the metagan D) and removing dead work (`_foreach_mul_(grads, 1.0)`, a per-parameter grad-norm loop).
7. Compatibility: no config key changes; `net_*.pth` pretrained loading is untouched; resuming an old
   `adan_sf` `.state` must be refused with the standard `ConfigError` hint (owner decision 4: old `.state`
   resume is best effort) via an optimizer state version recorded by `Phase`.

Priority for the owner (speed, reliability, memory, training efficiency): O-01/O-02 (learning efficiency
and stability of the default optimizer of 168 sections) > O-06 (resume safety of the change) > O-04/O-05
(memory and dead work, free with O-01) > O-03/O-11 (the A/B that decides the final recurrence and defaults)
> the P3 items.

## 1. F01 in depth

### 1.1 What the current `adan_sf` computes (schedule-free on), per step t >= 1

Notation: code betas `b1, b2, b3` (0.98, 0.92, 0.99); `g_t` the gradient at the live weights `y_t`;
`lr_t = lr * min(t / warmup, 1)`; `bc_i = 1 - b_i^t`; `w_t = t^r * lr_max^weight_lr_power` with
`lr_max = max_s lr_s * sqrt(bc3_s)`; `c_t = w_t / sum_{s<=t} w_s` (c_1 = 1).

```
m_t = b1 m_{t-1} + (1-b1) g_t                       adan_sf.py:43-44
d_t = g_t - g_{t-1}            (d_1 = 0)            :41        (neg_pre_grad holds -g_{t-1})
v_t = b2 v_{t-1} + (1-b2) d_t                       :46-47
u_t = g_t + b2 d_t                                  :49-50
n_t = b3 n_{t-1} + (1-b3) u_t^2                     :51-54
D_t = sqrt(n_t) / sqrt(bc3) + eps                   :56-58
y  <- (1 - lr_t*wd) * y                             :60   decay at y ...
y  <- (1-c_t) y + c_t z                             :66   ... erased at t=1 (c_1 = 1: y <- z = theta_0)
y  <- y - lr_t * bc1 * (1-c_t) * m_t / D_t          :63,67  multiplies by bc1 (Adan divides)
y  <- y - lr_t * (b2/bc2) * (1-c_t) * v_t / D_t     :62,68
z  <- z - lr_t * g_t                                :70   raw gradient: no preconditioning, no decay
```

Consequences, all reproduced (`out_f01.txt`):

| probe | current adan_sf | adamw_sf / sound SF-Adan |
|---|---|---|
| p = 1, g = +1, lr 0.01, wd 0: y after step 1 | **1.0000000000** (no move) | 0.9900000095 |
| same, z after step 1 | 0.9900000095 | 0.9900000095 |
| same, eval x = y + (1 - 1/b1)(z - y) | **1.0002040863 (uphill)** | 0.9900000095 |
| g = 0, lr 0.1, wd 0.1, one step: y / x / z | 1.0 / 1.0 / 1.0 (decay lost) | 0.999 / 0.999 / 0.999 |

Why it misbehaves: in schedule-free the three sequences must satisfy `y = (1-b1) z + b1 x` with `x` the
weighted average of `z`, and `z` must take the *same* preconditioned direction that moves `y`. Here `z`
takes an SGD step with the raw gradient and `y` takes an Adan step, so `y`, `x`, `z` drift apart. For
quadratics with curvature a and lr with `lr * a > 2` the raw `z` step is unstable regardless of the
preconditioner, and every step lerps `y` toward that exploding `z`:

| optimizer (`out_toy.txt`) | ill-cond quadratic (cond 1e3, lr 1e-2, warmup 50): loss @100 | final (2000) |
|---|---:|---:|
| adan_sf (current) | **4.6e13** | **5.1e29 (diverged)** |
| sound SF-Adan A / B | 1292 / 1291 | 4.5e-6 / 2.4e-26 |
| adamw_sf | 1220 | 9.7e-5 |
| adan, AdamW | 27 / 173 | 5.9e-15 / 2.0e-6 |

In SR runs `Phase.step` clips the total gradient norm to 1.0 (`image.py:71`, default `grad_clip = true`),
so the raw `z` step is at most `lr` in norm per update: `z` stays near `theta_0` plus a slow SGD drift,
and the live `y` is lerped toward it with weight `c_t ~ 1/t`. The optimizer then behaves like Adan at
`lr*(1-c_t)*bc1` with a persistent pull back toward the initialisation; that is why historical runs
learned (acceptance.md: esrgan live-y PSNR 26.2 dB at 2000 updates) and why it is slower than it should
be. The eval weights `x` are only an extrapolation `y - 0.0204 (z - y)` of that; acceptance.md shows SF x
scoring within +-0.4 dB of y, consistent with x and y being almost the same point.

Quantified lag on the toys (lower is better; `x@` = loss at the eval weights):

| problem | adan_sf (current) y@100 / final_x | SF-Adan B y@100 / final_x | adamw_sf y@100 / final_x |
|---|---|---|---|
| MLP regression, lr 1e-3, warmup 100, wd 0.01, 3000 steps | 2.71 / 0.121 | 1.07 / 0.194 | 0.670 / 0.056 |
| conv SR x2, lr 1e-3, 1500 steps | 0.596 / 0.077 | 0.275 / 0.059 | 0.228 / 0.065 |
| conv SR x2, lr 5e-3, 1500 steps | 0.268 / 0.067 | 0.152 / 0.036 | 0.190 / 0.049 |

At step 100 the current recurrence is 2-2.5x behind on every stochastic toy. (On the deterministic
cond-1 quadratic it wins, because there raw GD on `z` is the ideal step; irrelevant for SR.)

### 1.2 Reference recurrences

(a) Schedule-free (Defazio et al. 2024, `schedulefree/adamw_schedulefree.py`), gradient at
`y_t = (1-b1) z_t + b1 x_t`:

```
U_t     = g_t / (sqrt(v_t) + eps) + wd * y_t          v_t = b2 v_{t-1} + (1-b2) g_t^2
gamma_t = lr * sched_t * sqrt(1 - b2^t)                (bias correction folded into the step)
z_{t+1} = z_t - gamma_t U_t
x_{t+1} = (1-c_{t+1}) x_t + c_{t+1} z_{t+1}            c = w_t / sum w,  w_t = t^r * lr_max^wlp
in place: y <- lerp(y, z_t, c);  y <- y + gamma_t (b1 (1-c) - 1) U_t;  z <- z - gamma_t U_t
eval:  x = y + (1 - 1/b1)(z - y);   train:  y = x + (1 - b1)(z - x)
```
`adamw_sf.py` is this code (checked line by line and numerically, §4).

(b) Official Adan (sail-sg, `adan.py`), code-beta convention:

```
m_t = b1 m + (1-b1) g_t;  v_t = b2 v + (1-b2)(g_t - g_{t-1});  n_t = b3 n + (1-b3)(g_t + b2 (g_t - g_{t-1}))^2
theta <- theta - lr * [ m_t/bc1 + (b2/bc2) v_t ] / (sqrt(n_t)/sqrt(bc3) + eps);  decay: *(1 - lr wd) (no_prox) or /(1 + lr wd)
```
`adan.py` is this code (§4). Note that when `b1 = b2`, `m_t + b2 v_t` is the EMA of the extrapolated
gradient `u_t = g_t + b2 (g_t - g_{t-1})`: Adan is "Adam on the Nesterov-extrapolated gradient", with
the second moment also taken on `u_t`.

### 1.3 Derivation of a sound schedule-free Adan

Schedule-free turns momentum into the `x`/`z` interpolation: SF-AdamW is Adam **without** its first
moment `m`, wrapped in the `x`/`y`/`z` recurrence; `b1` is reused as the interpolation weight. The same
surgery on Adan: drop `m` (its role is taken by the SF average), keep what makes Adan different from
Adam, i.e. the extrapolated gradient `u_t` in the second moment and optionally the difference EMA `v_t`
in the numerator, keep Adan's bias corrections where a moment survives, put decoupled weight decay into
the direction at `y` (as the reference does) so that it reaches `z` and `x` consistently, and keep the
warmup/`lr_max`/`weight_lr_power` averaging weights.

Per step (`g_t` at `y_t`, `g_0 := g_1`):

```
d_t = g_t - g_{t-1}
v_t = b2 v_{t-1} + (1-b2) d_t                          [variant B only]
u_t = g_t + b2 d_t
n_t = b3 n_{t-1} + (1-b3) u_t^2
gamma_t = lr * sched_t * sqrt(1 - b3^t)                 (sched_t = min(t/warmup, 1))
numerator:  A (minimal)  u_t                             B (full)  g_t + (b2 / (1 - b2^t)) v_t
U_t = numerator / (sqrt(n_t) + eps) + wd * y_t
z <- z - gamma_t U_t
x <- (1-c_t) x + c_t z          (c_t = w_t / sum w,  w_t = t^r * lr_max^wlp,  lr_max = max gamma_s)
y <- (1-b1) z + b1 x            in place:  y <- lerp(y, z_old, c_t);  y <- y + gamma_t (b1 (1-c_t) - 1) U_t
```

Properties (all verified in `out_f01.txt` / `out_betas.txt`):
- t = 1: `c_1 = 1` gives `x_1 = z_1 = y_1 = theta_0 - gamma_1 U_1`: a full preconditioned step downhill
  (probe: max|y-z| = max|x-z| = 0 after step 1; scalar case 0.99 for all variants).
- Weight decay survives step 1 (0.999 for lr 0.1, wd 0.1, zero grad) and acts as in SF-AdamW afterwards
  (200 zero-grad steps, lr 1e-3, wd 0.01: 0.99918 vs adamw_sf 0.99913; plain Adan 0.99800). SF decay is
  weaker than decoupled decay by construction because `x` averages `z`; same as the reference.
- `b2 = 0`: variant A and B reduce to SF-AdamW with `(b1, b3)`: the `sfadan-B (.9,0,.99)` row equals the
  `adamw_sf b1=.9 b2=.99` row to every printed digit on all three toys (0.03362 / 0.04925, 0.064 /
  0.06353, 0.05565 / 0.0478).
- eval/train swap formula is unchanged (`x = y + (1 - 1/b1)(z - y)`), so `tests/test_resume.py::sf_x`,
  `eval_weights()` and the export path need no change.
- The variant D = "keep `m`" (the G4.M2 consensus formula `U = (m/bc1 + (b2/bc2) v)/denom + wd y`) is
  consistent as a recurrence but is double momentum (`b1` EMA with ~50-step memory on top of the SF
  average). Toys: MLP final 0.50 (B: 0.19), conv SR 0.48 (B: 0.057), conv SR lr 5e-3 0.66 = no learning.
  Reject.

Which numerator? Attribution on the toys (`out_betas.txt`, all SF, lr 1e-3 unless stated):

| variant | MLP final_y | conv SR final_y | conv SR lr 5e-3 final_y |
|---|---:|---:|---:|
| adamw_sf (0.9, 0.99) == SF-Adan b2=0, b1 0.9 | 0.034 | 0.064 | 0.056 |
| adamw_sf (0.98, 0.99) == SF-Adan b2=0, b1 0.98 | 0.068 | **0.043** | **0.035** |
| SF-Adan B (0.98, 0.92, 0.99) | 0.262 | 0.057 | 0.035 |
| SF-Adan B (0.9, 0.92, 0.99) | 0.231 | 0.069 | 0.061 |
| SF-Adan A (0.9, 0.92, 0.99) | 0.233 | 0.092 | 0.084 |
| SF-Adan B (0.9, 0.92, 0.999) | 0.211 | 0.064 | 0.057 |

The difference term (`b2 > 0`) never helps here and hurts on the mini-batch MLP by 4-7x. The reason is
structural: with mini-batches `g_t - g_{t-1}` is dominated by batch noise, not curvature; Adan's own
gains were reported for large-batch ViT/BERT training. SR training (random crops, batch 8) is noisier
than these toys. `b1 = 0.98` beats 0.9 on the conv toys and loses on the MLP. Toys cannot settle this
for SR: §6 makes both an arm of the A/B. The implementation should therefore support `b2 = 0` cheaply
(no `v`/`g_prev` buffers when the term is off) so that the two arms are the same code path.

## 2. Proposed implementation (O-01)

State per parameter: `z`, `exp_avg_sq` (n), and when `b2 > 0` also `neg_pre_grad` (-g_{t-1}) and, for
variant B, `exp_avg_diff` (v). 2 / 3 / 4 buffers instead of 5. Non-SF mode keeps the plain Adan kernel
(4 buffers) with warmup; it is bit-identical to `adan(no_prox=True)` today and stays so.

Foreach sketch for the SF branch (B; A drops the two `v` lines and the copy/add before the div):

```python
# grads already clipped by Phase; clip only when max_grad_norm > 0 (tensor), never _foreach_mul_(grads, 1.0)
torch._foreach_add_(neg, grads)                                    # neg = d = g - g_prev
torch._foreach_mul_(v, b2); torch._foreach_add_(v, neg, alpha=1 - b2)        # [B]
torch._foreach_mul_(neg, b2); torch._foreach_add_(neg, grads)      # neg = u = g + b2 d
torch._foreach_mul_(n, b3); torch._foreach_addcmul_(n, neg, neg, value=1 - b3)
denom = torch._foreach_sqrt(n); torch._foreach_add_(denom, eps)    # the only temporary (1x params)
torch._foreach_copy_(neg, grads); torch._foreach_add_(neg, v, alpha=b2 / bc2)  # [B] numerator
torch._foreach_div_(neg, denom)                                    # neg = U
if wd: torch._foreach_add_(neg, params, alpha=wd)                  # decay at y
torch._foreach_lerp_(params, z, weight=c)
torch._foreach_add_(params, neg, alpha=lr * (b1 * (1 - c) - 1))    # lr = group lr * sched * sqrt(bc3)
torch._foreach_add_(z, neg, alpha=-lr)
torch._foreach_copy_(neg, grads); torch._foreach_neg_(neg)         # -g_t for the next step
```

Per step: A = 13 foreach ops (+1 with wd), B = 17 (+1), current = 20; `b2 = 0` path = 9 (= adamw_sf).
`lr_max` / `weight` / `ckp1` stay Python floats (no host sync); `step`, `weight_sum`, `lr_max` stay group
scalars so `state_dict()` keeps its shape. `eval()`/`train()` keep the lerp formulas, but as one
`torch._foreach_lerp_` per group (O-08).

State-dict policy (O-06): add `STATE_VERSION = 2` class attribute on `adan_sf`; `Phase.state_dict()`
records `"version": getattr(type(opt), "STATE_VERSION", 0)` next to `"type"`/`"sf"`;
`Phase.load_state_dict` compares it like the type (`ConfigError` with the existing
"start a new run with pretrain_network_*" hint). `adan_sf.__setstate__` additionally
`setdefault("sf_version", 1)` per group and raises on 1 when `schedule_free`, so a bare
`torch.load`/`load_state_dict` outside Phase cannot silently continue either. Old `net_g/net_d .pth`
loading as pretrained weights is not affected (weights only). No legacy kernel: owner decision 4 makes
old `.state` resume best effort, and keeping the wrong recurrence alive as a mode would keep 168
templates on it by accident; the clean error plus "resume with pretrain_network_g" is the honest path.

Behaviour change: **yes**, objective/dynamics of the default optimizer of 168 sections (§6 A/B before
it ships as default). Config compatibility: none (same keys). Checkpoint compatibility: old `adan_sf`
`.state` refused; `.pth` unaffected.

Test plan (replaces `test_adan_sf_warmup_scales_z_step`, which asserts the raw-gradient `z` step, i.e.
the bug):
1. Reference trajectory: explicit `x`/`z` per-tensor implementation (as `sf_variants.SFAdanBase`) vs the
   foreach kernel, 300 steps with warmup 20 and wd 0.01 on random grads: `assert_close` on `y`, `z`, `x`.
2. `b2 = 0` equals `adamw_sf(betas=(b1, b3))` bit-for-bit on the same trajectory.
3. Step 1: `x_1 == y_1 == z_1`, and for a scalar `p = 1, g = +1` all three equal `1 - lr/(1 + eps)` (with
   `bc3` folded: `lr * sqrt(1-b3) / (sqrt(1-b3) |g| + eps)`); eval moves downhill.
4. Zero gradient + wd: one step gives `(1 - lr*wd)` on `y`, `x`, `z`.
5. Warmup: the step-1 `z` delta with `warmup_steps = 10` is 1/10 of warmup 0 (same assertion as today,
   against the preconditioned direction).
6. eval/train round trip and `state_dict` round trip (existing tests, unchanged).
7. Old state refused: a state dict whose group lacks `sf_version` raises on `load_state_dict`, and a
   `Phase` state with `version` missing or 1 raises `ConfigError` naming the optimizer.
8. `tests/test_resume.py` (b) and `tests/test_model.py` SF tests pass unchanged (x formula unchanged).

## 3. Line-by-line checks of the other two optimizers (O-13, verified, nothing to fix)

`adamw_sf.py` vs `schedulefree/adamw_schedulefree.py`: `sched = (k+1)/warmup` (same 1-based rule), `lr =
lr*sched*sqrt(bc2)`, `lr_max`, `weight = (k+1)^r * lr_max^wlp`, `ckp1`, the "Not in train mode" check
position, lazy `z`/`exp_avg_sq` init, `exp_avg_sq` update, `denom = sqrt + eps`, grad normalised in place,
decay added at `y`, `lerp` then `add` with `lr*(b1*(1-ckp1)-1)`, `z -= lr*grad`, the loop path, and
`eval`/`train` lerps all match. Only the defaults differ (lr 8e-4 / betas (0.9, 0.99) vs 2.5e-3 /
(0.9, 0.999)), which is configuration. Numeric check against an explicit `x`/`z` reference, 300 noisy
steps, warmup 20, wd 0.01: max |x - x_ref| = 1.43e-6, max |z - z_ref| = 6e-8 (fp32 rounding of the
in-place `y` form), foreach and loop paths identical. Edge: `step()` updates `weight_sum`/`lr_max`
before raising "Not in train mode", exactly like the reference; `Phase` never steps in eval mode.

`adan.py` vs official `sail-sg/Adan`: kernels identical (m, v, n updates, `denom`, `step_size`,
`no_prox` branch, `neg_pre_grad` reuse, step-1 re-init of `neg_pre_grad`). Differences: neosr keeps the
clip factor as a 0-dim tensor (official `.item()`s it: a host sync per step), and defaults `no_prox=True`,
`weight_decay=0.02`, `lr=5e-4` (official False / 0 / 1e-3). The docstring still says `no_prox`
"(default: False)" and `adan_sf`'s docstring says betas `(0.98, 0.92, 0.9955)` while the code has
`0.99` (O-10). Probe: foreach == single tensor (0.0 difference, both `no_prox`), and
`adan_sf(schedule_free=False)` == `adan(no_prox=True)` (0.0) after 100 steps.

Competitor `traiNNer/optimizers/adanschedulefree_optim.py`: a different unofficial variant. It moves `z`
with the Adan direction (better than neosr's raw gradient) but keeps `m` under SF (variant D, see §1.3),
applies decay to `y` *after* the lerp and not to `z` (so `x` and `z` never see decay), uses the `y`
coefficient `(1-c)` instead of `(1-b1(1-c))` (the `y = (1-b1) z + b1 x` relation does not hold after
step 1), `.item()`s the clip factor (host sync per step when `max_grad_norm > 0`) and defaults
`warmup_steps=10000`. Nothing to port. Its `adan_optim.py` is the official Adan with `.item()`.

## 4. Speed and memory (O-04, O-05, O-07, O-08, O-12)

Counts are top-level aten ops per `step()` for 120 tensors / 6M params on CPU (`out_ops.txt`); one
`aten::_foreach_*` is one multi-tensor-apply launch group on CUDA (1-3 kernels each for this size).
CPU wall times are not reported: the machine was shared with other lanes and foreach falls back to
per-tensor loops on CPU; GPU timing belongs to the profiling lane.

| optimizer | foreach/top-level ops per step | state buffers / param | temporaries |
|---|---:|---:|---|
| adan_sf (SF) | 20 | **5** | 1x params (`_foreach_sqrt`) |
| adan_sf (SF, `max_grad_norm>0`) | 20 + **3 per tensor** (pow, sum, add_) | 5 | 2x |
| adan_sf (no SF) / adan | 18 | 4 | 1x |
| adamw_sf | 9 | 2 | 1x |
| torch AdamW foreach | ~13 (+ CPU-only `.item()` on step counters) | 2 | 1x |
| torch AdamW fused | **2** (`_foreach_add_` step, `_fused_adamw_`) | 2 | 0 |
| proposed SF-Adan A / B / `b2=0` | 13-14 / 17-18 / 9 | 3 / 4 / 2 | 1x |

Dead work in the current kernel (source-verified, `adan_sf.py:35`, `adan.py:92`): `_foreach_mul_(grads,
1.0)` runs a full read-modify-write pass over every gradient when `max_grad_norm == 0` (the templates'
case); the profiler shows the kernel is launched. `_foreach_zero_` + `_foreach_add_(alpha=-1)`
(`adan_sf.py:76-77`) is two passes where `_foreach_copy_` + `_foreach_neg_` is two cheaper ones, or one
if the sign convention of `neg_pre_grad` is flipped to `+g_{t-1}` and `d` is formed with `_foreach_sub`.

State memory for template archs (fp32, trainable params, `out_params.txt`):

| arch | tensors | params | 5 bufs (now) | 4 | 3 (A) | 2 (b2=0) |
|---|---:|---:|---:|---:|---:|---:|
| compact | 53 | 0.62M | 12 MiB | 9 | 7 | 5 |
| span | 164 | 1.84M | 35 | 28 | 21 | 14 |
| esrgan | 702 | 16.7M | 318 | 255 | 191 | 127 |
| dat_m | 1702 | 14.8M | 282 | 226 | 169 | 113 |
| atd | 1024 | 20.3M | 386 | 309 | 232 | 155 |
| hat_m | 862 | 20.8M | 396 | 317 | 238 | 158 |
| metagan (D, also adan_sf) | 154 | 6.0M | 115 | 92 | 69 | 46 |

ATD + metagan: 501 MiB today, 301 MiB with variant A, 201 MiB with `b2=0`: 200-300 MiB saved on a 24 GiB
card, i.e. 1-2 more batch items at patch 32 for a transformer, nothing dramatic. Bandwidth estimate for
ATD: 20 passes x ~160 MiB traffic ~ 3.2 GB -> ~3.5 ms per update on a 4090 (~900 GB/s); 14 passes ~2.4
ms. Against a 300-500 ms transformer update this is < 0.5%: **the speed argument for O-01 is small and
honest; its value is correctness and memory.**

`max_grad_norm` (O-07): `adan_sf.py:195-210`, `adan.py:189-204` compute the global norm with a Python
loop `global_grad_norm.add_(grad.pow(2).sum())`: 3 launches per tensor (ATD: ~3000 launches, several ms)
and it duplicates `Phase.step`'s `get_total_norm(foreach=True)` + `clip_grads_with_norm_` that already
ran on the same gradients (`phase.py:92-96`). No template sets it. Delete the option (two fewer code
paths, test `test_max_grad_norm_*` goes) or, if kept, use `torch._foreach_norm` (probe: 485 vs 1564
aten ops on CPU for 120 tensors). Behaviour change: none for shipped configs.

`eval()`/`train()` (O-08): per-parameter Python loops with `p.data.lerp_` (`adan_sf.py:165-186`,
`adamw_sf.py:88-109`): 1 launch per tensor, twice per validation (ATD: ~2000 launches), plus
`eval_weights()` clones every parameter one by one (`image.py:302`: `[p.detach().clone() for p in
net.parameters()]`, another ~1000 launches and a 1x-params transient). Validation-only cost; one
`torch._foreach_lerp_` per group and `torch._foreach_copy_` into a preallocated flat buffer fix it.
Low priority, free to do while the kernel is rewritten.

bf16/fp16 parameters: all states are `zeros_like(p)` so they follow the parameter dtype (probe: bf16
params give bf16 `exp_avg_sq`, `z`, ...). neosr keeps fp32 master parameters under autocast
(`Phase.autocast`), so this is not reachable from a config; note it in the docstring, no code change.

Fused AdamW (O-12, hypothesis): `torch.optim.AdamW(fused=True)` is one kernel per step and compatible
with `Phase.step`'s `scaler.unscale_` -> `scaler.step` sequence (the scaler takes the normal path after
`unscale_`). For the 6 `adamw` templates it is a one-line default in `OPTIMIZERS`/`build_optimizer`
when the parameters are on CUDA; expected effect < 1% of an update, so measure on the GPU before
claiming it. `torch.compile` of the custom foreach step would fuse the 13-18 passes into 2-3 but adds
compile time and a recompile risk on every parameter-set change; not worth it for < 0.5%.

## 5. Trainer integration (O-06, O-09, F19)

- `Phase.state_dict` records `type`, `sf`, `names`, `state`; `load_state_dict` compares type/sf/names/
  schedulers then `self.opt.load_state_dict(o["state"])`. torch replaces every group hyper-parameter
  with the saved one (`optimizer.py:update_group`): probe `out_state.txt`: a saved `lr = 5e-3` overrides
  a live `lr = 1e-3` silently, and a key absent from the saved group disappears (so `__setstate__`
  `setdefault` is the right hook for a version). **F19, optimizer part: confirmed.** Cheap fix inside
  this lane: in `Phase.load_state_dict`, diff `{lr, betas, weight_decay, warmup_steps, eps}` of the
  current groups against `o["state"]["param_groups"]` with the existing `dict_diff` helper and log one
  warning line "net_g: resumed lr 1e-3 overrides configured 1e-4" (the loss/accumulate/data part of F19
  belongs to the trainer lane).
- `is_schedule_free(opt)` reads `opt.defaults["schedule_free"]`, not the groups (`phase.py:33`). Probe:
  a group override `schedule_free=True` with defaults False gives `sf = False` while `opt.eval()` moves
  the parameters: exports would be `y`, not `x`, and the resume identity would be wrong. Not reachable
  from a config (`build_optimizer` passes one flat parameter list) but a one-line hardening:
  `any(g["schedule_free"] for g in opt.param_groups)`, or forbid per-group SF overrides in
  `add_param_group`.
- `eval_weights()` and `export_state_dicts()` are correct for SF (snapshot, `opt.eval()`, copy back,
  restore `train_mode` flags; never `opt.train()` because a second lerp is not a bit inverse). The
  `x` formula does not change with O-01, so `tests/test_resume.py::sf_x` stays the oracle.
- GAN implication of O-01 (hypothesis, important for the A/B): the D phase uses the same broken
  `adan_sf` (`lr 1e-4`, warmup 600). Today D is handicapped the same way as G (pulled toward its
  initialisation). acceptance.md's "the GAN is balanced" was measured with both handicapped. After the fix
  both learn faster; the balance can shift, so the A/B must log `out_d_real/out_d_fake`, `l_g_gan`,
  `grad_norm_d` and be ready to lower D's lr (5e-5) or keep D on plain `adan`.

## 6. Recommended defaults and the A/B (O-11)

Evidence available: 168 template sections use `adan_sf` with G `lr 1e-3` (64 sections; 84 use `1e-4`,
which are the D sections, 16 `5e-4`, 4 `8e-4`), `betas (0.98, 0.92, 0.99)`, `wd 0.01`, warmup 1600 (G)
/ 600 (D); the competitor ships AdamW `2e-4` G / `1e-4`-`5e-5` D, no warmup, `(0.9, 0.99)` (41 sections)
and its own SF-Adan in 8 sections at `1e-4`; acceptance.md ran `adan_sf 1e-3` G + `1e-4` D for 2000
updates with clip 1.0 and no skipped step. Literature: Defazio et al. recommend warmup, `b1 0.9-0.98`
(higher for long runs), `weight_lr_power 2`, `r 0`, and found the best SF-AdamW lr 1-10x the AdamW lr;
Adan's paper used `(0.98, 0.92, 0.99)`, `wd 0.02`, `eps 1e-8`, large batches.

Recommendation for the first arm = *change only the recurrence*, keep every template value: G `lr 1e-3`,
`betas (0.98, 0.92, 0.99)`, `wd 0.01`, `eps 1e-8`, `warmup 1600`; D `lr 1e-4`, `warmup 600`. Reasons:
(i) a recurrence fix and an lr change together cannot be attributed; (ii) SF tolerates lr above AdamW's
because `y` sits near the average `x` and only moves by `lr*(1-b1+b1 c)` per step (~2% of the `z` step
at `b1 0.98`); (iii) the clip to 1.0 bounds the `z` step. Candidate second arms: `b2 = 0` (drop the
difference term: the toys' best and 2 buffers), G `lr 5e-4`, `b1 0.9`. Not recommended: changing `eps`,
`wd` or `weight_lr_power`.

A/B protocol (dataset `/home/korvin/apps/data`: GT 10607 PNG 256x256, LQ x4 64x64, val 7 pairs):
1. Two archs: `compact` (fast, 0.6M) and `dat_m` or `atd` (transformer, 15-20M). G-only first (L1 +
   mssim, `ema 0.999`, `ema_power 0.75`, no D) to isolate the optimizer; then the shipped template with
   metagan + GAN for the balance check.
2. Arms: (0) current `adan_sf` (baseline, from the same seed), (1) fixed B with template values,
   (2) fixed with `b2 = 0` (== SF-AdamW `(0.98, 0.99)`), (3) `schedule_free = false` (plain Adan +
   warmup, same lr), (4) competitor-style control: `adamw 2e-4`, `(0.9, 0.99)`, MultiStepLR.
3. 20k updates, `manual_seed` fixed, 2 seeds for the compact runs; bf16 AMP, batch 8, patch 64 (compact)
   / 32 (transformer); identical data order.
4. Record per 1k updates: val PSNR/SSIM on live `y`, SF `x`, EMA (three weight sets, as acceptance.md
   did), `l_g_total`, `grad_norm_g/d`, `skipped_*`, and for the GAN run `out_d_real/fake`, `l_g_gan`;
   it/s and `torch.cuda.max_memory_allocated` once at steady state.
5. Decision rule: adopt the fixed recurrence as default if arm 1 or 2 reaches arm 0's 20k PSNR earlier and
   is >= it at 20k on both archs; pick A/B/`b2=0` by the PSNR gap exceeding the seed spread; otherwise
   ship the fix with `schedule_free = false` in templates and keep SF opt-in.

## 7. Findings (ordered by value for the owner's priorities)

| id | sev | category | title | verified | behaviour change | effort |
|---|---|---|---|---|---|---|
| O-01 | P1 | learning | `adan_sf` SF recurrence is unsound: `z` takes the raw gradient, `y` is pulled toward it, step 1 is a no-op, eval is uphill, decay erased; diverges on ill-conditioned problems, 2-2.5x slower on stochastic toys | reproduced | yes: dynamics/objective of the default optimizer (A/B §6); old `adan_sf` `.state` refused; `.pth` unaffected | M |
| O-02 | P1 | learning | The planned G4.M2 formula (keep Adan's `m` under SF) is double momentum and learns badly (conv toy 0.48 vs 0.057; no learning at lr 5e-3): do not implement it | reproduced | n/a (decision) | S |
| O-06 | P1 | reliability | Resuming an old `adan_sf` state onto the new kernel must fail loudly: `STATE_VERSION` in `Phase` state + `__setstate__` default; otherwise saved `z`/moments silently continue under different semantics | source-verified (group-merge probe) | checkpoint: old SF `.state` -> `ConfigError` | S |
| O-04 | P2 | memory | 5 state buffers/param -> 3 (A), 4 (B) or 2 (`b2=0`): ATD+metagan 501 MiB -> 301 / 201 MiB | measured (param counts) | none beyond O-01 | S (in O-01) |
| O-05 | P2 | speed | 20 foreach passes -> 13-18; `_foreach_mul_(grads, 1.0)` is a wasted full pass; zero+add -> copy+neg | measured (op counts) | none | S (in O-01) |
| O-03 | P2 | learning | The Nesterov difference term (`b2 = 0.92`) brings nothing on the toys and hurts on mini-batch MLP 4-7x; must be an A/B arm; keep `b2 = 0` cheap (no extra buffers) | reproduced (toys), hypothesis for SR | yes (A/B arm) | S |
| O-11 | P2 | learning | Defaults: first arm keeps template values (G 1e-3 / D 1e-4, (0.98, 0.92, 0.99), wd 0.01, warmup 1600/600); the D was handicapped too, so GAN balance must be re-checked; A/B protocol in §6 | hypothesis + literature | yes (A/B) | L (GPU time) |
| O-07 | P3 | speed | Optimizer-level `max_grad_norm`: 3 launches per tensor (ATD ~3000) and duplicates `Phase` clipping; delete or use `_foreach_norm` | measured | none (no template uses it) | S |
| O-10 | P3 | reliability | F19 optimizer part: saved group hyper-parameters silently override the config on resume; log a `dict_diff` of lr/betas/wd/warmup in `Phase.load_state_dict`; fix docstring defaults (`no_prox`, `adan_sf` betas) | reproduced | none | S |
| O-08 | P3 | speed | `eval()`/`train()` and `eval_weights()` snapshot are per-tensor Python loops (ATD ~3000 launches per validation); use foreach | source-verified | none | S |
| O-09 | P3 | reliability | `is_schedule_free` reads `defaults`, not the effective group flag: a per-group SF override would export `y` and mis-identify the resume; unreachable from config today | reproduced (direct construction) | none | S |
| O-12 | P3 | speed | `adamw`/`adam` table entries: `fused=True` on CUDA (2 ops vs ~13); `torch.compile` of the custom step not worth it (< 0.5% of an update) | hypothesis (CPU op counts only) | none | S |
| O-13 | — | correctness | `adamw_sf` == reference (1.4e-6 after 300 steps), `adan` == official kernel, `adan_sf(no SF)` == `adan(no_prox=True)`: nothing to fix | reproduced | none | — |

## 8. Codex verdicts

| item | verdict | note |
|---|---|---|
| F01 | confirmed (and extended) | First-step no-op on `y`, uphill eval (1.0002040863), erased step-1 decay, `bc1` multiply, `z` with raw gradient: all reproduced. Added: divergence on an ill-conditioned quadratic, 2-2.5x lag on stochastic toys, mechanism in real runs (clip keeps `z` near init), and a refutation of the G4.M2 consensus formula. Codex's "do not merely reverse the evaluation sign" is right. |
| F19 (optimizer part) | partial -> confirmed for optimizer hyper-parameters | Saved groups override lr/betas/wd/warmup silently (torch `update_group`); the loss/accumulate/data part is outside this lane. |
| analyze.md "168 sections use adan_sf" | confirmed | `grep` over `options/`: 168 `type = "adan_sf"`, all `schedule_free = true`, 167 with `(0.98, 0.92, 0.99)`. |
| PLAN.md S.2 (warmup for plain optimizers) | not-checked | Out of scope; with O-01 the SF optimizers remain the only ones with warmup. |

## 9. Competitor material used

`traiNNer/optimizers/adanschedulefree_optim.py` (different unsound SF-Adan, see §3; not portable),
`traiNNer/optimizers/adan_optim.py` (official Adan with `.item()`), `adamwschedulefree_optim.py`
(reference copy), `options/train/**` (AdamW 2e-4/1e-4, no warmup; 8 SF-Adan sections at 1e-4),
`experiments/*` (no optimizer comparison logs found). Nothing in its optimizers is better than neosr's;
its stability mechanisms are the automations the owner rejects and were not considered.
