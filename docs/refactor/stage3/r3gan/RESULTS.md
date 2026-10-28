# R3GAN in neosr: A/B results (stage 3 R4-R6)

Brief: `TASK_R3GAN.md`; design and acceptance gate: `DESIGN.md` section 4. Runs, configs and the harness live outside git
in the worktree's `experiments/r3gan_ab/` (`harness/` = runner with per-image metric capture, config generator, launcher,
parity and visual scripts; `cfg/`; `runs/<arm>/{train,metrics,per_image}.jsonl`, `experiments/<arm>/{models,visual}`).

## 0. Round 0: data and judges (fixed before any A/B run)
| item | value |
|---|---|
| pool | the owner's 10,607 x4 pairs (`/home/korvin/apps/data/{GT,LQ}`, GT 256 / LQ 64 tiles, names are content hashes) |
| source recovery | content hashing: every 16x16 block of every tile looked up against all tiles' stride-16 grid blocks (exact integer hashes, non-flat blocks): 488 overlapping pairs; plus 18 confident edge adjacencies (mutual best border continuation, ratio < 0.3); union -> 10,144 source groups (385 multi-tile groups hold 848 tiles, largest 8) |
| split (by group, seed 2026) | `split/dev.txt` 300 tiles (290 groups), `split/conf.txt` 300 tiles (290 groups, untouched until the final check), `split/train.txt` 10,007 tiles; excluded from every run incl. the pretrain |
| duplicates vs `val8` | none (no shared block between any tile and the 8 val8 GT photos) |
| limit | tiles of one photograph that share no pixels and no border are not recoverable from content; tile names carry no source. Residual same-photo leakage between train and held-out is possible but cannot inflate one arm over another (all arms share the split) |
| `val8` | 8 real photos, 512 px GT / 128 px LQ: domain sanity set |
| visual set (`split/visual.txt`) | 16 dev tiles + 4 real-LQ 512 px crops of `val8/Anna Ralphs - SHOOTS - 0068.jpg` (nose skin, upper lip, tongue, fingernail). The owner's data are close-up photos of people, so the design's foliage/sky/text categories were replaced by what the domain has: eyes, faces with hair, teeth/lips, eyebrow, hair, pores/freckles, tattoo lettering, rings, patterned fabric (non-explicit tiles only) |
| judges | TOPIQ-FR `dataset = "pipal"` (higher better), LPIPS alex (lower), DISTS (lower), PSNR (fidelity sanity); none is a training loss in any arm (FDL is dinov2); EMA weights every 2k updates |

## 1. Predeclared margins and decision rules (written before round 1)
- Unit of comparison: the 300 dev tiles at 20k GAN updates (EMA weights); per-image paired differences arm - reference,
  95 % CI by a bootstrap over the 290 source groups (10,000 resamples). val8 is reported, never decisive.
- Practical margins: TOPIQ-pipal +0.005, LPIPS -0.003, DISTS not worse than +0.002, PSNR not below A0 - 0.3 dB.
  "Better than X" on a metric = the mean paired difference beats the margin and its CI excludes 0; after round 3 the margin
  is the larger of the above and X's seed-to-seed spread on that metric (|seed 7 - seed 2| of the dev means).
- Round 1 -> round 2: the best R arm is the one with the largest TOPIQ-pipal gain over A1 among R arms that pass PSNR
  and stability; ties (CIs overlapping) go to LPIPS, then to cost. If no R arm beats A1, round 2 still runs on the best R
  (the question then is whether any setting makes the GAN useful) and the outcome is likely "inconclusive" or "no".
- Stability (diagnostic, not a quality score): no skipped updates, finite `r1`/`r2`/penalties, `out_d_margin` without
  monotone runaway, G's adversarial gradient share non-zero (`scripts/term_grads.py` at 0/10k/20k).
- Cost: `scripts/bench.py` median ms/update within +30 % and peak VRAM within +1.5 GB of A0, else the owner decides.
- Outcomes allowed: replace, do not replace, inconclusive. The owner's visual sign-off and an 80k+ run are required before
  any template default changes (proposed, not run here).
