# neosr-dev baseline smoke report (unmodified code; no tracked files modified)

Scratch: /tmp/claude-1000/-home-korvin-apps-neosr-dev/8bf983be-1946-4899-9a0e-9ccad5a13242/scratchpad/smoke/ (configs *.toml, logs *.log, arch_results.json, arch.log)
Run stopped early on coordinator request: test.py was NOT run, R2/R3 not completed as specified (see below).

## 1. Architecture smoke (99 registered names)
Harness note: `import neosr` / `neosr.archs` requires `-opt <config>` in sys.argv (neosr/utils/rng.py:11 -> parse_options raises "Didn't get a config!"). I set sys.argv. Archs only register after importing each *_arch.py (build_network does it lazily); I imported all, 0 import failures.
Result: 85 OK, 14 fail (4 are harness misuse, 10 real/likely real).

| name | error | frame |
|---|---|---|
| dat_2, dat_m, dat_s | ValueError: Expected more than 1 value per channel when training, got [1,22,1,1] (BatchNorm in train mode, batch=1) | neosr/archs/dat_arch.py:563 (via forward 1067/1053/845/754) |
| esc_fp | ValueError: DySample can only be enabled on esc_real. (default kwargs invalid) | neosr/archs/esc_arch.py:497 |
| grformer, grformer_medium, grformer_large | RuntimeError: clamp max tensor on cpu vs cuda | neosr/archs/grformer_arch.py:237 |
| krgn | RuntimeError: tensors on cuda:0 and cpu | neosr/archs/krgn_arch.py:18 |
| hit_srf_medium / hit_srf_large | CUDA OOM, tried 6.00 / 9.61 GiB for 64x64 input (GPU was shared with other jobs, ~6 GB free at that moment, but allocation is huge anyway) | neosr/archs/hitsrf_arch.py:322 / :312 |
| omnisr | KeyError: 'upsampling' with default kwargs | neosr/archs/omnisr_arch.py:1057 |
| eqrsrgan | TypeError: forward() missing 'dist' (harness: needs extra input) | n/a (harness) |
| ResNetFeatureExtractor | output is dict, no .shape (harness; forward works) | harness |
| VGGFeatureExtractor | needs `layer_name_list` ctor arg (harness) | harness |

Side effect: resnet34 weights were downloaded to ~/.cache/torch/hub (outside the repo).

## 2. Training runs (train.py, 60 iters, bs16, BF16 unless noted)
Config deviations: print_freq=30 instead of 10 (see anomaly A1); R5/R7 had wavelet_guided removed (A3); R6 needed extra top-level keys (A4).

| Run | Status | Notes |
|---|---|---|
| R1 (smoke_r1, esrgan+metagan+gan+mssim+cons+ldl+fdl dinov2) | INTERMITTENT. Reached 60 and exit 0 in 1 of ~4 attempts (others: NaN abort at iter 1). Successful run: l_g_gan 6.7 (it30) -> 0.54 (it60), l_d_real 2.6->0.80, l_d_total 2.8->6.1, l_g_total 23.1->14.2; ckpts net_g/net_d_30, _60, _latest, training_states 30,60.state, vis for 7 val imgs. No warnings. Speed: ~0.93 it/s cumulative avg at 60 incl ~47 s first iter; steady state approx 3.5 it/s (from dr1 debug run). dinov2 loaded fine (no network failure) so fdl not removed. |
| R1 original-as-specified (print_freq=10, name smoke_r1) | exit 0 but STOPPED at iter 10 by speed guard (A1); only net_*_10.pth | |
| R2 (accumulate=2) | NOT completed as specified. 4 attempts: NaN abort (x4) and 1 OOM (GPU shared with other jobs). One attempt (c1) started OK but speed guard "Interrupted" at printed iter 15 (exit 0), ckpt net_g/net_d_30 only. print line at iter 15 with print_freq=30 and checkpoint named 30 -> iteration numbering inconsistent with accumulation (was investigating, A5). Loss values at that print sane (l_g_gan 2.6, l_d_total 5.3). |
| R3 (fp16 + GradScaler) | FAIL: NaN abort in 4/4 attempts (exit 1). Debug-named run (per-iter log) lasted 24 iters, l_g_total grew 210->640, l_d_total ~5000 (D losses huge), then NaN. |
| R4 (resume R1, --auto_resume, total_iter=90) | PASS. Resumed from training_states/60.state ("Resuming training from epoch 0, iteration 60"), continued to 90, saved net_*_90 + latest, exit 0. Minor: eta printed as "-1 day, 23:59:58"; perf shows 0.528 it/s. No warnings. |
| R5 (compact, mssim+vgg_perceptual, no GAN) | As specified (wavelet_guided=true kept): exit 1 immediately: "Wavelet-Guided requires GAN" (A3). With wavelet_guided removed: PASS, 60 iters, exit 0, 2.0 it/s cumulative, l_g_percep ~0.94-0.96, l_g_total ~1.0, no NaN, ckpt net_g_30/60/latest. |
| R6 (OTF compact + metagan etc.) | As shipped: FAIL KeyError 'poisson_scale_range' (A4). With 4 keys also at top level (jpeg_range(2), poisson_scale_range(2)): PASS 60 iters, exit 0, 1.78 it/s, l_g_gan 0.30->0.20, l_d_total 0.80->0.73 (sane). A fully flattened [degradations] variant fails in dataset (sinc_prob None). |
| R7 (R5 + [val] tile=64) | PASS (with wavelet removed): 60 iters, exit 0, 1.57 it/s, validation at 30/60 ran without error, 7 vis images. No metrics configured so no validation numbers; tile path coverage is not verified beyond "no exception". |
| test.py | NOT RUN (stopped on request). options/test_compact.toml and test_esrgan.toml exist. |

No "warn" lines in any training log. NaN-like values only as the NaN aborts described.

## 3. Tracebacks (deduplicated)
1. NaN abort (R1, R2, R3): `ValueError: NaN found, aborting training...` raised at neosr/models/image.py:725 (closure), called from image.py:741 (optimize_parameters), train.py:552. In R1/R2 it occurs at iteration 1 (no log line ever printed; confirmed with 4 debug-named runs "dnan1-4": 2 died at iter 1 with no iter line, 2 ran to 60). In R3 (fp16) at ~iter 24.
2. OOM during backward (R2 attempt a2): torch.OutOfMemoryError, 196 MiB, ~12 GB was held by other processes on the shared GPU. Not code-attributable.
3. R5 as shipped: no traceback; `ERROR: Wavelet-Guided requires GAN.` then exit 1.
4. R6 as shipped: `KeyError: 'poisson_scale_range'` at neosr/models/otf.py:146 (also :207 for the '2' variant), from train.py:551 feed_data.
5. R6 flattened: `TypeError: '<' not supported between float and NoneType` at neosr/data/otf_dataset.py:199 (sinc_prob missing from dataset opts).
6. Arch failures: see table in section 1.

## 4. Anomalies
- A1 Speed guard: train.py:581-584 exits with status 0 ("Interrupted, saving latest models.") if cumulative it/s < 0.5 at a print step, unless "debug" is in the run name. The first iteration takes ~47 s (startup/warmup), so with print_freq=10 or 20 the guard always triggers silently (exit 0, did not reach total_iter). Names containing "debug" bypass it but options.py:265-267 then force val_freq=8, print_freq=1, save_checkpoint_freq=8. Also `iter_time/accumulate` makes accumulate>1 runs trigger it more easily.
- A2 Nondeterministic NaN at iteration 1 in R1/R2 (about 50-70% of starts, no manual_seed), always in fp16. R1 D outputs very large at start (out_d_real 358, l_d 1e3 in a debug run), suggesting metagan logits / bce under bf16 unstable early. Root cause not isolated.
- A3 options/train_esrgan2.toml-derived no-GAN configs: wavelet_guided=true requires GAN, so a non-GAN run must drop it.
- A4 train_compact_otf.toml: train.py:131-132 copies [degradations] only into dataset opts, but neosr/models/otf.py reads poisson_scale_range(2)/jpeg_range(2) from top-level opt, so the shipped config can't run as is.
- A5 With accumulate=2, the printed "iter" (15) and checkpoint name (30) disagree; log says "Accumulated batches: 32" (it is batch*accumulate, mislabelled "Accumulated batches"); with accumulate unset it prints "Accumulated batches: 16".
- A6 eta can print negative ("-1 day, 23:59:58").
- A7 GPU is shared with other processes (usage 2-12 GB observed), which causes OOM/timing noise.
- A8 "dnan" runs: debug-named copies of R1 (smoke_debug_nan1..4, print_freq=1) used only to find at which iteration the NaN abort occurs.
- A9 Stray file arch_results.json was written into repo root by my script and moved to scratch immediately; git status afterwards showed only pre-existing changes (.codex/*, AGENTS.md deleted, info.txt, install.sh, options/train_esrgan2.toml).
