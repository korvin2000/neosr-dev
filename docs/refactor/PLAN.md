# Implementation plan (v1 — core phases are refined when design/T1..T4, T6 land)

Status values: `todo` | `wip` | `review` | `done` (commit hash). Every step: implement -> verify (checks listed) -> log in
PROGRESS.md -> commit. Finding ids refer to `analysis/consensus/*.md` (G1..G8, F) — consult them for exact fixes.
Competitor rule: before changing an area, read `analysis/competitor/H*.md` (when available) and the same-purpose files in
/home/korvin/apps/traiNNer-redux-dev; adapt, never copy as-is.

## Phase 0 — groundwork
| step | content | files | status |
|---|---|---|---|
| 0.1 | Delete legacy models/datasets/archs (owner decision) + docs skeleton | models/default,simple_otf,eqr_otf,pancam_otf; data/*; archs/pancam,eqrsrgan | done (43db06a) |
| 0.2 | Loss safety-net tests (tests/test_losses.py, conftest with temporary argv hack) | tests/ | done (155f677) |

## Phase L — leaf fixes (parallel, disjoint files; no structural changes)
| step | scope (consensus ids) | files owned | checks | status |
|---|---|---|---|---|
| L1 | Loss numerics: G5.1 ldl eps + locals, G5.2 ncc per-(N,C), G5.3 ms-ssim clamp, G5.4 delete PatchLoss/IPK + vgg_arch load fix, G5.5 consistency L* + msswd loop, G5.6 dists freeze, G5.7 buffers, G5.M2 vgg criterion, G5.M3 cosim gate, G5.M5 buffers/devices, G5.M7 chc reduction, G5.M8 fdl lower, G7.M3/M6 dists metric path; delete resnet_perceptual_loss + archs/resnet_arch | neosr/losses/* (except __init__.py, kl_loss.py), neosr/archs/vgg_arch.py, archs/resnet_arch.py, tests/ | tests green without xfails, no global default device | done (155f677) |
| L2 | Optimizer internals: G4.1 adan_sf warmup lr to kernel, G4.3 adamw_win/adamw_sf validation, G4.M4 drop **kwargs, G4.M5 delete restart_opt, G4.M6 enable_grad closure, G4.M6b SF gating, G4.M9 no .item() | neosr/optimizers/* (soap_sf deletion waits for core step) | unit probes: warmup changes update magnitude; SF train/eval round trip | done (14e3b2b) |
| L3 | Data pipeline: G6.1 loader args (DDP), G6.2 record_stream, G6.4 augmentations, G6.6 queue sizing helper, G6.7/M7/M9 paths + meta_info, G6.8 pin/persistent + CPU sampler, G6.M1 crop size, G6.M3 noise view, G6.M4 blur_kernel_size keys, G6.M5 dead helpers, G6.M6 worker seeding, G6.M10 rand_bbox, F.10 decode errors | neosr/data/* | dataset smoke (paired/otf/single), loader iteration, worker seeds distinct | done (9360403) |
| L4 | Metrics/utils/archs: G7.2 cached metric nets, G7.M7 crop checks, F.2/G7.M1 tensor2img, G1.M2/M3 resgan conv3 + simam, G8.6 DySample pin_memory, G8.M1 dctlsa toml, G8.M3 test_scnet + personal paths | neosr/metrics/*, neosr/utils/img_util.py, neosr/archs/{resgan,arch_util(pin only),ea2fpn}_arch.py, options/train_dctlsa.toml, options/test_scnet.toml, options/test_aether.toml | metric values equal before/after; val speed | done (52137a8) |

Phase L launched 2026-10-20 as workflow `phase-L-leaf-fixes` (impl t3 -> Codex review -> fix pass); status -> review when it returns.

## Phase A — architecture fixes (parallel, disjoint arch files; state_dict keys must not change)
| step | scope | files owned | status |
|---|---|---|---|
| A1 | H4 #1 FlexNet attention no-grad/layout/window, H4 #10 reparam lifecycle | flexnet, span, spanplus archs + tests/test_archs_a1.py | todo (WIP patch, unverified) |
| A2 | H4 #3 window padding, H4 #11 mask dtype/exit/mean buffer | hat, drct, swinir, swinirdense, dat archs + tests/test_archs_a2.py | todo (WIP patch, unverified) |
| A3 | H4 #12 EIMN, smoke: grformer/krgn device, esc_fp/omnisr defaults, hit_srf OOM check | eimn, grformer, krgn, esc, omnisr, hitsrf archs + tests/test_archs_a3.py | todo (WIP patch, unverified) |
| A4 | later/optional: H4 #17 spectral_norm parametrization (needs D checkpoint key remap), #15 shared UniUpsample, #18 compact learn_residual, #19 capability flags | - | todo |

## Phase C — core restructure (sequenced; refined from design/T1..T4, T6)
| step | content | depends | status |
|---|---|---|---|
| C1 | Config/construction (T3): parse once, no import-time argv (net_opt/rng), scale injection into archs/losses, RNG ownership, exceptions instead of sys.exit, default-device policy | L* | todo |
| C2 | Training step + loss pipeline (T1 + T6): LossContext, losses pull own inputs, GAN G/D terms (G1.*), one accumulation counter (G2.*), AMP policy, NaN/inf detection without per-iter sync, EMA gating, SAM/ECO handling | C1 | todo |
| C3 | Model structure + checkpoints (T2): base/image/otf split, live+EMA+scaler+counters in state, atomic saves, compile/DDP unwrapping, resume semantics (G3.*), validation fixes (G7.1, G7.4, G7.M8) | C2 | todo |
| C4 | train.py/test.py/convert.py loop rewrite: counters, epochs (G2.M1/M7), kill-switch removal, resume errors, logging (G2.*, G3.2, G7.3) | C3 | todo |
| C5 | Aether registration + QAT end to end (T5; resolve canonical source: T5 says aether_arch v4, G8.4 says aether_core v5) | C3 | todo |
| C6 | Performance pass (T4): syncs, layouts, compile mode, tiled val on GPU, profiler before/after | C4 | todo |
| C7 | Stability acceptance (T6): NaN rate 0 over N seeds, D logits bounded, 2k-iter bf16+fp16 runs image+otf, resume equivalence; template defaults | C4 | todo |
| C8 | Docs: readme/options templates cleanup, final review (Codex + Fable), remove temporary test hacks | all | todo |

G5.M4 DISTS normalisation (loss + metric) applied in 155f677 (DECISIONS #6).
Open 3-model tie-break: canonical aether source (T5: aether_arch v4 vs G8.4: aether_core v5) — resolve at C5.

## Session-1 end state (2026-10-20) — start here next session
1. Phase A: unverified WIP saved as `wip/phaseA_archs_unverified.patch` (+ `wip/test_archs_a*.py`); the worktree was reset clean.
   Resume: `git apply docs/refactor/wip/phaseA_archs_unverified.patch`, copy the tests into tests/, then for each group A1-A3:
   verify (state_dict keys unchanged vs HEAD, forward equality, tests) -> Codex review -> fix -> commit. The workflow was stopped
   mid-run, so some groups may lack their Codex review/fix pass — treat all three as unreviewed.
2. Designs: T1, T2, T3 final in `design/`; T4 has only the synthesis draft (`design/drafts/T4_feature_triage_perf.md`, no
   critique/final); T5 final; T6 stability playbook NOT produced — evidence in `analysis/stability/raw_summary.md` (neosr
   instability is mechanical: competitor survives neosr-like hyperparameters, current neosr NaNs in 4/6 seeds).
   Next: finalize T4 + T6 with Opus (Codex as reviewer; Fable only if a strategic fork appears), then refine Phase C rows above.
3. NaN root cause: Codex analysis in `analysis/nan/codex.md`; the empirical t4 run was stopped before reporting
   (its logs were in the session scratchpad and are not preserved beyond `analysis/stability/raw_summary.md`).
4. Then Phase C (C1 -> C8) per the table above; Codex reviews every step.

## Phase C execution plan v2 (session 2, from design/T1-T3 §6/§9/§5; T5 for aether)
Rule: agents implement against CURRENT HEAD (designs were written at 55f0cdd; Phase L already landed parts — skip what is done).
Each lane: Opus implements -> Codex gpt-6-astra reviews -> Opus fix pass -> main-thread review -> commit.
| wave | lanes (disjoint files) | status |
|---|---|---|
| W0 | seed `neosr/config.py` (ConfigError/ConfigWarning) by main thread; **A-verify** (apply wip patch: A1-A3 + T1-A4 swinirdense reentrant); **T3-0** foundation (config.py, runtime.py, cli.py, utils/factory.py, dist_util.py, tests config/runtime/imports/registry/hygiene); **T1-A3** models/phase.py (Phase, Ema, LossMeter) + tests; **T2-A1** models/checkpoint.py + tests; **T2-A2** models/tiling.py + tests; **T2-A4** utils/stop.py + tests | todo |
| W1 | **L-A** archs R1 recipe (batch 0: archs/__init__ table + build_network(scale=) + arch_util; batches 1-6) incl. call-site `scale=` one-liners; **L-B** metrics table; **L-C** data/RNG atomic switch (utils/rng.py, data/**, models/otf.py:20); **T1-A1** losses/base.py + losses/__init__ builder (LOSSES table) | todo |
| W2 | **T1-A2** gan_loss + AdvView; **T1-B1** loss kernel recipe (3 workers) incl. FDL seeded projections; **T2-A3** data/otf_degradation.py (Degrader, PairPool); **T2-B1** phase state_dict | todo |
| W3 | **C1** integration (one agent): models/base.py, image.py, otf.py, models/__init__ (T2 C1 + T1 B2 + T3 L-E + T5 QAT slot) | todo |
| W4 | **D1** entry points: train.py, test.py, convert.py, utils/logger.py, utils/misc.py, utils/__init__.py, prefetch reset (T2 D1 + T1 B3 + T3 L-E) | todo |
| W5 | verification (T2 E1-E4, T3 I-1 smoke + DDP gloo + errors, T1 D step tests) + T3-Z cleanup; then C5 aether/QAT, C6 perf, C7 stability acceptance, C8 docs | todo |
