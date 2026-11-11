# A/B harness

Every 8k/20k A/B of a training change uses these three tools. Runs, configs and data stay outside git (`experiments/`).

| file | does |
|---|---|
| `run_ab.py` | trains one arm (a TOML) through train.py's own machinery and writes `train.jsonl`, `metrics.jsonl`, `per_image.jsonl` and, after the final validation, `done.json` into the arm's run folder |
| `queue.sh` | runs arms from a queue file (`A,B` = a pair side by side, `A` = alone), one line at a time; a line starts only if it can end by `DEADLINE`; an optional `LOCK` (flock) makes it the GPU's only owner |
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

8k runs are a screen. A default changes only by the owner's decision, and a WIN on a GAN default stays provisional until a longer run (`next_tasks.md`, `STATUS.md`).

## Running it (Linux workstation)
1. Write one TOML per arm into a session folder `experiments/<session>/cfg/`, each a text edit of the template. `name`, `total_iter` and `save_checkpoint_freq` set the run. `experiments/s9_ab/mkcfg.py` is an example. To freeze the code an arm trains with, put the checkout path in `cfg/<arm>.code`.
2. Queue the arms (two 8k arms side by side take ~29 min on the RTX 4090; one ~18 min):
   ```sh
   printf 'B0,B0s2\nB1,B2\n' > experiments/<session>/queue.txt
   CFG=experiments/<session>/cfg RUNS=experiments/<session>/runs QUEUE=experiments/<session>/queue.txt \
     DEADLINE=16:30 LOCK=<gpu.lock> MEMORY_MAX=12G nohup scripts/ab/queue.sh > experiments/<session>/queue.log 2>&1 &
   ```
   One arm by hand: `python scripts/ab/run_ab.py <repo> <cfg>/<arm>.toml <runs>/<arm>`. Add `--resume` to continue the arm's last state.
3. Verdicts:
   ```sh
   python scripts/ab/analyze.py --runs experiments/<session>/runs --data <data> --seed-ref B0s2 --iter 8000 B0 B1 B2
   ```
   `--data` takes either a folder in the standard layout or a JSON file
   `{"sources": ..., "groups": ..., "exclude": ..., "manifest": {"dev": "<gt dir>"}}`, with paths relative to the file. The standard layout is `sources.json`, optional `groups_eval.json` and `exclude.txt`, and one `<dataset>_gt/` per dataset. The manifest guard stops on an arm that scored a missing or duplicate image.

## Where things live on the Linux machine (gitignored)
| what | path |
|---|---|
| data (`train/dev/conf/val8` `_gt`/`_lq`, `sources.json`, `groups_eval.json`, `exclude.txt`) | `/home/korvin/apps/neosr-dev/.worktrees/refactor/experiments/r3gan_ab/data` (session folders link it as `data`) |
| pretrain G | `.../r3gan_ab/runs/P0_pretrain/pretrain_g.pth` |
| reference configs (20k; `A0_template`, `A1_nogan` = no D, `R1_patch_g1` = R3GAN) | `.../r3gan_ab/cfg/` |
| session 9 (8k arms, gate notes, R3GAN-specific scripts) | `/home/korvin/apps/neosr-dev/experiments/s9_ab/` |
