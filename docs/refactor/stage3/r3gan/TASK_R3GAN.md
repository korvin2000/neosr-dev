# R3GAN track: implementation and A/B brief (for Opus 5.5, reasoning xhigh)

Goal: implement `DESIGN.md` in this worktree (`/home/korvin/apps/neosr-dev/.worktrees/refactor`, branch `refactor/stabilize`,
Phase A landed through fa07061), then run the A/B on the owner's data and report. Read first, in this order: `docs/refactor/README.md`
(resume protocol), `stage3/r3gan/DESIGN.md` (the decisions; binding), `stage3/r3gan/neosr_fit.md` (sections 1.3, 2, 3, 4, 5, 7:
the specs, numbers and the 20 tests), `stage3/r3gan/audit.md` sections 3, 6, 7 (reference facts, fixes), `stage3/r3gan/codex.md`
(independent review). Owner rules as in `stage3/TASK_OPUS.md` (binding): fast, stable, reliable, compact; no automation or
damping; explicit tables, no registries; losses read `Context`, no if-chains in the loop; defaults preserved (`bce` + metagan
templates untouched); spandrel-compatible keys; exact resume; commit per verified item on `refactor/stabilize`, never push;
Codex reviews every diff (`Agent` with `subagent_type: codex:codex-rescue`, brief starts `--model gpt-6-astra --effort high`,
"read-only: advise only, do not edit files", hand it the diff and ask what is wrong); log each item in `stage3/PROGRESS.md`.
CPU tests: `.venv/bin/python -m pytest tests/<file> -q`, memory capped at 4 GB, one job at a time. GPU (RTX 4090, 24 GB):
one job at a time, never concurrently with another training run, memory-capped as the stage-2 A/Bs were; check `nvidia-smi`
before each launch.

## Items, in order
**R1 Discriminator** `neosr/archs/r3gan_arch.py` + `ARCHS` rows `r3gan`, `r3gan_scalar128`, `r3gan_scalar256` (DESIGN 1.1;
`neosr_fit.md` 2.1-2.6). Pure torch; group width 16; FIR as a stride-2 depthwise conv with a non-persistent `[1,3,3,1]/8`
outer-product kernel; in-place biased leaky ReLU 0.2; fix-up init exactly as `neosr_fit.md` 2.1 (forward gain 1); zero-init
`lin3`; heads `patch` (1x1 to one logit per position) and `scalar` (exact 4x4 depthwise basis + 1x1, `ValueError` naming the
right GT size otherwise); 4-D output; `widths`, `group_width`, `expansion`, `blocks`, `head` kwargs. Tests 1-5 of
`neosr_fit.md` section 7 (golden keys and param counts 0.87 M / 1.21 M / 1.55 M, zero-init and linearity at init, FIR
equivalence incl. channels_last, `gradgradcheck` on a tiny config incl. the in-place activation, compiled-vs-eager). Add the
arch to the export tables as a D (no ONNX) the way the other Ds are listed.

**R2 Loss path** (`neosr/losses/gan_loss.py`, `neosr/losses/base.py`, `neosr/losses/__init__.py` RESERVED keys; DESIGN 1.2;
`neosr_fit.md` 3.1-3.5). (1) `reshape(len(out), -1)` instead of `flatten(1)`. (2) Rademacher per-position probe exactly as
`neosr_fit.md` 3.3 (`eps` from the device RNG, one draw per regularised update, `P = 1` draws nothing: bitwise today's
penalty). (3) Penalty pass under the phase autocast when it is bf16 with fp32 leaves and fp32 accumulation; fp32 eager
otherwise; `penalty_fp32: bool = False` forces fp32. (4) `gamma_decay`, `gamma_iters` cosine schedule as a pure function of
`ctx.it` from the first GAN update. (5) Logs: raw `r1`, `r2` (per-sample squared norms before gamma), weighted `l_d_r1`,
`l_d_r2`, `l_d_reg` = their sum, the effective gamma. (6) Fail-fast as DESIGN 1.2 item 6: `validate_train` refuses `dunet` with a
gamma > 0 by name; the first D call refuses a batch-coupled D with a gamma > 0 (`ConfigError`); no runtime probe; fp16 runs keep
the fp32 penalty. (7) Validation as DESIGN 1.2 item 7; effective gammas computed before deciding to run the penalty (zero skips
the graph and the draw). Draw policy: fresh signs per sample, position, side and micro-batch on regularised updates (DESIGN
1.2 item 2); single-GPU resume exactness via the saved device RNG; multi-rank is not bitwise (document, as the trainer already
says for data order). Tests 6-18 of `neosr_fit.md` section 7, including: pairing order (batch and position), penalty
on known functions (disjoint receptive fields exact for any draw; exhaustive sign patterns for P = 2, 3 on overlapping ones),
exhaustive parameter-gradient tests of the estimator (expectation over all sign patterns equals the exact per-position
penalty's gradient; P = 2, 3), P = 1 bitwise today's under the same precision path and no RNG consumption, gammas-0 bitwise,
lazy k-scaling, schedule values and the zero-gamma skip, bf16 vs fp32 parity of the estimator with identical signs (replaces
the fp32 pin in `tests/test_s2_losses.py:506`), MATH-SDPA double backward incl. the documented dunet refusal, `(B,)` output, resume parity
across a schedule and RNG draws, DDP 2-rank gloo with the patch head. Record the two behaviour changes of this opt-in path
(map penalty meaning, penalty dtype) in `docs/refactor/DECISIONS.md`.

**R3 Recipe arm and docs**: `options/train_realplksr_r3gan.toml` (DESIGN 1.3, gamma 1 per position, Adam (0, 0.99), `reg_every
= 1`, `wavelet_guided = false`, `pretrain_network_g` pointing at the round-0 pretrain export) validated by the template sweep
test; a `nafnet` 1x variant is not needed now. README/guide: one paragraph on `r3gan` and the rpgan + R1/R2 settings.

**R4 GPU qualification** (before any A/B): `scripts/bench.py` on `train_realplksr_r3gan.toml` vs `train_realplksr.toml` (ms/update
median for ordinary and regularised updates, warm-up, peak VRAM, eager and `compile = true`); bf16 vs `penalty_fp32 = true`
with identical probe signs on 10 real batches at init and at a trained checkpoint, comparing penalty values and the
penalty-only D-parameter gradients (cosine, norm ratio, per-layer outliers); one-probe gradient variance and clip incidence at
the trained checkpoint; bf16 channels_last double backward finite; a short controlled training comparison (2k GAN updates,
bf16 vs fp32 penalty, same seed: metrics, margins, raw R1/R2). bf16 stays the recipe default only if this passes; otherwise
the template sets `penalty_fp32 = true`. If `r3gan` exceeds +30 % ms/update or
+1.5 GB vs the template, also bench `widths = (16, 32, 64, 128, 256)` and `group_width = 32` and pick the cheapest that passes;
report all numbers in `PROGRESS.md`.

**R5 Round 0 (E0-lite data)**: a fixed held-out split **by source photograph** of the 10,607 tiles (`/home/korvin/apps/data/{GT,LQ}`;
recover the source from the tile names or by content hashing, check duplicates against train and `val8`): ~300 development
tiles plus an untouched confirmation set, as `meta_info` lists excluded from every run including the pretrain; paired
per-image metric differences with source-grouped uncertainty in the reports; `val8` as a domain sanity set, a visual set of 16 held-out
tiles (foliage, hair, fabric, text, faces, sky) + 4 fixed 512 px crops of `val8/Anna Ralphs - SHOOTS - 0068.jpg` as real LQ for a
no-reference visual check. Judges: TOPIQ-FR `dataset = "pipal"`, LPIPS (`better = "lower"`), DISTS, PSNR; EMA weights every 2k.
Then the shared non-GAN pretrain: the realplksr template without `gan_opt`, 20k updates, export `pretrain_g.pth`. The stage-2
harness (`/tmp/claude-1000/-home-korvin-apps-neosr-dev/c1b40b02-f454-4498-9d89-be0afa4e2973/scratchpad/ab/{launch.py,...}`)
may be reused; if a runner is kept in the repo, it is one small `scripts/ab.py` (config overrides + launch + metric table), no more.

**R6 A/B rounds 1-2 and the second seed** exactly as DESIGN section 4 (arms A0, A1, R1-R5 with A1 matched to the GAN arms in
objective, optimizer reset, EMA init and budget; then round 2 on the best R; then the top two arms + A0 + A1 with a second
seed; margins predeclared before the runs; "inconclusive" allowed). Per arm record: metric table at 2k steps (held-out + val8), `bench.py` ms/update and
peak VRAM, `out_d_margin`, raw `r1`/`r2`, `grad_norm_d`, `clipped_d`, skips, `scripts/term_grads.py` shares at 0/10k/20k (separate process, on a copy), and
the visual set at 5k/10k/20k saved under `experiments/<arm>/visual/`; equal-update and equal-wall-clock tables. Decide by the acceptance gate in DESIGN section 4; the
owner's visual sign-off and the 80k+ long run are the owner's (propose, do not run). Write `stage3/r3gan/RESULTS.md`: per arm
numbers, the gate, the recommendation (replace the template GAN or not, and with which settings), and what stays unverified.

## Out of scope
Phase B items (OTF realism, i-LN, KD/ECO), template default changes before the gate passes, beta2 warm-up and per-net clip unless
round-1 logs call for them (then as DESIGN 3 says, small and recorded), DDP GPU runs, anything in the do-not-port list.

## Done means
R1-R3 committed one by one with Codex-reviewed diffs and green scoped tests, full suite green once, R4 numbers and R5 data
recorded, R6 rounds 1-2 + second seed run and `RESULTS.md` written, `PROGRESS.md` current, and a final report: per item the
commit, tests, Codex verdict, open points; per arm the headline numbers; the recommendation; anything skipped and why.
