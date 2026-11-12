# A/B harness

Every 8k/20k A/B of a training change uses these three tools. Runs, configs and data stay outside git (`experiments/`).

| file | does |
|---|---|
| `run_ab.py` | trains one arm (a TOML) through train.py's own machinery and writes `train.jsonl`, `metrics.jsonl`, `per_image.jsonl` and, after the final validation, `done.json` into the arm's run folder: `seconds`, `update`, `run_peak_gib` (the process's VRAM peak: the start-up transient, not the working set) and `steady_mem_gib` (the median window peak of `train.jsonl`; `null` without `mem_gib` rows). Runs finished before 2026-11-12 name the run peak `peak_mem_gib`; `analyze.py` reads it as `run_peak_gib` |
| `queue.sh` | runs arms from a queue file (`A,B` = a pair side by side, `A` = alone), one line at a time (blank and `#` lines skipped; it ends when the file is empty); a line starts only if it can end by `DEADLINE` (`HH:MM`: its next occurrence); an optional `LOCK` (flock) makes it the GPU's only owner. A line leaves the file only after a check (one or two distinct arm names, each with a `<CFG>/<arm>.toml`): any other line stops the queue with an error and stays in the file. The queue flocks the queue file itself, so a second queue on it refuses to start: one queue file per runner, edited in place (an editor that saves by replacing the file drops the lock) |
| `analyze.py` | dataset means, paired per-image differences against a reference with a 95 % CI over source groups, the gate's verdict per arm, training diagnostics (D margin, R1/R2, grad norms, clip share, skips) |

## Protocol (stage 3, `docs/refactor/stage3/r3gan/RESULTS.md` sections 0-1)
| item | value |
|---|---|
| data | the owner's 10,607 x4 pairs (GT 256 / LQ 64 tiles), split by recovered source group (seed 2026): train 10,007, dev 300, conf 300 (untouched until a final check) |
| evaluated | 219 dev tiles in 211 groups: a fuzzy leak check removed held-out tiles that are crops of train photos (`exclude.txt`); `val8` (8 real photos) is a sanity set, never decisive |
| start | the 20k non-GAN pretrain `P0_pretrain` (G from it; D and optimizers fresh), the shipped realplksr + metagan template (`A0_template.toml`), seeded, EMA weights |
| judges | TOPIQ-FR pipal (higher is better), LPIPS alex (lower), DISTS (lower), PSNR (fidelity, reported) |
| noise | a seed pair of the reference (`B0`, `B0s2`) measures run-to-run spread |

Gate, per metric against the reference:
- **floor** = max(seed spread, 0.05 dB for PSNR, 1 % of the reference mean for TOPIQ/LPIPS/DISTS). The seed spread is \|reference - its second seed\| of the means over the compared images.
- **better / worse** only when the mean paired difference is beyond the floor in that direction and its 95 % CI excludes 0. The CI comes from a bootstrap over source groups (10,000 resamples), so tiles of one photo are not counted as independent.
- **verdict** over the perceptual metrics: **LOSS** if TOPIQ, LPIPS or DISTS is worse; **WIN** if at least 2 of the 3 are better and none is worse; otherwise **TIE**. PSNR is shown with its own status but does not decide.

**The D10 gate** (`docs/refactor/DECISIONS.md`, 2026-11-12; binding for arms started after it) replaces the one-checkpoint floor: `--window FROM:TO` compares per-image means over the validated updates in the window (screen 4000:8000, decision 12000:20000), and `--null X/Y,...` names same-config run pairs whose RMS delta sets **tau** = max(3 x null RMS, 0.05 dB PSNR or 1 % of the reference mean; TOPIQ without the 1 % term). The reference may pool runs of one config (`A0+B0+B0s2`). `--report-only dists` reports a metric the arms train on without gating it; WIN then needs every gated metric (both, of two) better. `LOSS (mixed)` = a gated metric worse and another better: not a clear loser. The diagnostics add `ms_update` (median over log windows) and `mem_gib` (the first window's start-up peak, the steady median, the maximum; `run_ab.py` logs it per window).
   ```sh
   python scripts/ab/analyze.py --runs <runs> --data <data> --window 12000:20000 --null A0/B0,B0/B0s2 --report-only dists A0+B0+B0s2 T_DISTS
   ```

8k runs are a screen. A default changes only by the owner's decision, and a WIN on a GAN default stays provisional until a longer run (`next_tasks.md`, `STATUS.md`).

## Compile parity (D2; `docs/refactor/DECISIONS.md` D10 rule 8)
`scripts/compile_parity.py` runs one training update from a saved state, each mode in a fresh process on one fixed batch (drawn once, `restore_rng()` before the update, `deterministic = true`; with `accumulate` > 1 every micro-batch of the window is that batch), and compares the G/D/EMA updates and the losses.

| mode | role |
|---|---|
| `eager` | the reference |
| `eager2` | eager again: the run-to-run floor |
| `compile` | `compile = true`, compared with `eager` |
| `eager_fp32` | `use_amp` off: the bf16 noise scale (vs `eager`) |
| `compile_fp32` | compiled without bf16, compared with `eager_fp32` |

**PASS** when, per net, rho = \|\|dp_compiled - dp_ref\|\| / \|\|dp_ref\|\| and every `l_*` loss's relative difference stay within `--tol` (default 2) times the bf16 noise scale (at least the floor), and every mode restored the same state bitwise; a noise scale that is not finite fails. R1/R2, D outputs and grad norms are reported, not gated. A D with Dropout/DropPath (metagan) draws inductor's random stream when compiled: a different noise sample, not a numeric bug, but it fails the verdict; `--fallback-random` makes the compiled modes draw eager's stream and so tests the numerics alone. Checked against the session-10 results: realplksr + unet and + R3GAN pass, metagan fails without `--fallback-random` (its D losses) and passes with it.
```sh
python scripts/compile_parity.py <cfg>/<arm>.toml <runs>/<arm>/experiments/<arm>/training_states/8000.state <scratch>/parity/<arm> \
  [--code <checkout>] [--modes eager,eager2,compile,eager_fp32,compile_fp32] [--fallback-random] [--tol 2]
```
The state is only read; everything else goes below the output folder (`batch.pt`, `<mode>/result.pt`, `<mode>/child.log`, the run folders). A mode whose result exists is not run again; `args.json` refuses another setting in the same folder: other arguments, or a config, state or checkout (HEAD and the diff of tracked files) changed since. Exit 0 = PASS, 1 = FAIL, no verdict or a failed mode. It is a GPU job: never next to a training run.

## Running it (Linux workstation)
1. Write one TOML per arm into a session folder `experiments/<session>/cfg/`, each a text edit of the template. `name`, `total_iter` and `save_checkpoint_freq` set the run. `experiments/s9_ab/mkcfg.py` is an example. To freeze the code an arm trains with, put the checkout path in `cfg/<arm>.code`.
2. Queue the arms. The deadline guard uses estimates, not measurements: `PAIR_MIN` 29 for two 8k arms side by side on the RTX 4090, `SINGLE_MIN` 18 for one.
   ```sh
   printf 'B0,B0s2\nB1,B2\n' > experiments/<session>/queue.txt
   CFG=experiments/<session>/cfg RUNS=experiments/<session>/runs QUEUE=experiments/<session>/queue.txt \
     DEADLINE=16:30 LOCK=<gpu.lock> MEMORY_MAX=12G nohup scripts/ab/queue.sh > experiments/<session>/queue.log 2>&1 &
   ```
   One arm by hand: `python scripts/ab/run_ab.py <repo> <cfg>/<arm>.toml <runs>/<arm>`. A fresh run empties the arm's logs. `--resume` continues from the last state and keeps the log events up to it; a final state whose validation was cut off is validated again. `done.json` is deleted at the start and written atomically at the end.
3. Verdicts:
   ```sh
   python scripts/ab/analyze.py --runs experiments/<session>/runs --data <data> --seed-ref B0s2 --iter 8000 B0 B1 B2
   ```
   `--data` takes either a folder in the standard layout or a JSON file
   `{"sources": ..., "groups": ..., "exclude": ..., "manifest": {"dev": "<gt dir>"}}`, with paths relative to the file. The standard layout is `sources.json`, optional `groups_eval.json` and `exclude.txt`, and one `<dataset>_gt/` per dataset. Image ids are the paths below `<dataset>_gt/` without suffix, as the trainer names them. The tool exits rather than weaken the gate when an input is missing: a seed reference without rows, a metric, the sources file, the manifest, or a second source group. `--independent` treats every image as its own group; use it only for unrelated images, e.g. `--dataset val8`. The manifest guard also stops on an arm that scored a missing or duplicate image.

## Where things live on the Linux machine (gitignored)
| what | path |
|---|---|
| data (`train/dev/conf/val8` `_gt`/`_lq`, `sources.json`, `groups_eval.json`, `exclude.txt`) | `/home/korvin/apps/neosr-dev/.worktrees/refactor/experiments/r3gan_ab/data` (session folders link it as `data`) |
| pretrain G | `.../r3gan_ab/runs/P0_pretrain/pretrain_g.pth` |
| reference configs (20k; `A0_template`, `A1_nogan` = no D, `R1_patch_g1` = R3GAN) | `.../r3gan_ab/cfg/` |
| session 9 (8k arms, gate notes, R3GAN-specific scripts) | `/home/korvin/apps/neosr-dev/experiments/s9_ab/` |
