# Stage 3, Phase A: implementation brief (for Opus 5.5, reasoning xhigh)

Goal: land the Phase A items of `docs/refactor/stage3/REVIEW.md` on branch `refactor/stabilize` in the worktree
`/home/korvin/apps/neosr-dev/.worktrees/refactor`. Every item keeps today's behaviour by default; nothing automatic; no GPU
experiments (those are the owner's Phase 0). Read first: `docs/refactor/README.md` (resume protocol), `stage3/REVIEW.md`,
`stage3/CONSENSUS.md`, then the lane sections named per item below. Keep `stage3/PROGRESS.md` current (one entry per item:
done, verified, decisions, dead ends).

## Rules (owner, binding)
- Fast, stable, reliable, compact code. No damping or automation of any kind. No new registries: extend the explicit tables
  (`LOSSES`, `ARCHS`, `_metric_builders`, `SCHEDULERS`, `config.SCHEMA`). Losses read inputs from `Context`; no `if`-chains in
  the training loop. State-dict keys spandrel-compatible; every old `net_*.pth` must still load; `.state` resume stays exact.
- Smallest coherent diff per item; reuse existing helpers (`atomic_save`, `prune_exports`, `read_weights`, `Loss.active`,
  `_lookup`/`construct`, `window_sdpa`, metric build-once/park-on-CPU path). No new hard dependency; `safetensors` is an
  optional import with a `ConfigError` hint.
- Work item by item: implement -> scoped tests (`.venv/bin/python -m pytest tests/<file> -q`, CPU, memory capped at 4 GB as
  prior lanes did, one job at a time) -> Codex review of the diff (`Agent` with `subagent_type: codex:codex-rescue`,
  read-only advise, model per CLAUDE.md; this project used `gpt-6-astra --effort high` today) -> fix -> commit on
  `refactor/stabilize` -> `PROGRESS.md`. Never push. Never start GPU training. Run the full suite once at the end.
- Templates: only comments/rows that are off by default (A3 judge rows); no default changes.

## Items, in order
**A1 Exports and loading** (`neosr/models/{checkpoint,base,image}.py`, `convert.py`, `test.py`, `neosr/cli.py`,
`neosr/archs/esrgan_arch.py`; lane `competitor/training_infra.md` 3.3, 3.4, 3.6, 3.12)
- Export payload `{"params": sd, "neosr": {"network_g": <table>, "scale", "update", "metric": {...}}}`, builtins only
  (`weights_only`-safe). `read_weights` keeps picking `params`, so old files and spandrel/chaiNNer are unaffected.
  `convert.py`/`test.py` default `-net`, `-s` and `network_g` kwargs from the file; explicit CLI values win; files without
  the entry fall back to the CLI as today.
- `val.save_best = "<metric key>"`: when `update_best` improves that metric, rank 0 writes
  `models/net_g_best_<dataset>_<metric>.pth` holding exactly the evaluated EMA/SF weights plus `update`/`value`, via
  `atomic_save`, after `validate()` returns (never nested in `eval_weights()`); the best record is part of the state so
  resume is correct; `prune_exports` already ignores non-numbered names (verify).
- `read_weights`: `.safetensors` via optional `safetensors.torch.load_file(..., device="cpu")`.
- Relaxed load: capped (10) missing/unexpected list + matched-keys percentage instead of one warning per key; strict
  default and the training-state guard unchanged.
- `esrgan_arch`: `_load_from_state_dict` remap of the old community `model.N...` layout (pattern: asid/ditn), with a test
  built from a synthesised old-layout state dict.

**A2 Scheduler** (`neosr/models/{base,image,phase}.py`, `neosr/config.py`; lane infra 3.1 design notes, 3.2; CONSENSUS Phase 1 #2)
- `train.scheduler.nets`: explicit optimizer targeting. Default = optimizers not in schedule-free mode, logged once; an
  explicit list may include them.
- Resume: unchanged scheduler config -> exact state restore (today). Changed config -> rebuild from config and replay to the
  saved step, with a loud warning naming old/new values; scheduler lists may change on resume (cooldown from any checkpoint).
- `validate_train`: `T_max` or last milestone < `total_iter` -> warning; range checks for new keys. No `wsd` row now (after E2).

**A3 Metrics** (`neosr/metrics/{calculate,topiq}.py`, new `neosr/metrics/lpips.py`, `tests/test_metrics.py`; lane
`competitor/losses_metrics.md` 2.2)
- `calculate_topiq(_pt)`: `dataset = "kadid" | "pipal"`; net cache keyed per dataset; strict load; default `kadid`
  unchanged; PIPAL weights `cfanet_fr_pipal_res50-69bbe5ba.pth` (URL in pyiqa `topiq_arch.py:28`, local checkout
  `/home/korvin/apps/IQA-PyTorch`; confirm the mirror filename before hard-coding).
- LPIPS v0.1 rewritten from the reference (~60 lines): torchvision AlexNet features (vgg optional), lin layers from the
  official v0.1 weights, `strict=True`, input `/255 -> [-1,1]`, ScalingLayer, unit-normalised features, spatial mean; one
  `_metric_builders` row, `better="lower"`, same build-once/park-on-CPU/`inference_mode` path. No per-image construction.
- Template metric rows (commented, off): name the judge that is not a training loss.

**A4 Losses** (`neosr/losses/{base,gan_loss,basic_loss,__init__}.py`, `tests/test_losses*.py`; lane losses_metrics sec. 3)
- Move `start_iter`/`ramp_iters` from `gan_loss` into `Loss` and add `end_iter`; the builder pops the keys for any term;
  `active()` plus a host-float factor in `compute()`; linear only; `gan_loss` behaviour and D gating unchanged (tests).
- `gan_type = "hinge"` next to `rpgan`: D `relu(1-r).mean() + relu(1+f).mean()`, G `-f.mean()`; no labels.
- `psnr_loss`: mean over images of `log10(mse_i + eps)` scaled to dB, fp32, `reduction` honoured. Only together with A7.

**A5 `scripts/bench.py`** (new; lane infra 3.9, 3.10; stage-2 harness notes in `stage2/analysis/gpu_profile.md`)
- Builds the trainer from a TOML as `train.py` does; reports median ms/update at steady state (warm-up and compile time
  reported separately), peak VRAM and host RAM, inference ms at a given size; flags for eager/compile, `compile_mode`
  (so `reduce-overhead`/`max-autotune` can be compared), bf16/fp16, N updates. CPU smoke test with a tiny arch only;
  the GPU runs are the owner's and never run concurrently with training.

**A6 RealPLKSR `layer_norm` + ICNR** (`neosr/archs/{realplksr_arch,arch_util}.py`, `tests/test_archs*.py`; lane
`competitor/archs.md` 2.1, 2.5)
- `layer_norm: bool = False` on `PLKBlock`/`realplksr`: channel LayerNorm before the block as `F.layer_norm` on the NHWC
  permute (zero-copy under channels_last, fp32 under autocast), post-branch `GroupNorm` -> `Identity` when on; keys
  `layer_norm.weight/bias` only when on (spandrel names, eps 1e-6). GN default untouched.
  Tests: key sets for both settings; init output std < 0.5 with LN; fp16 forward finite with large activations.
- `icnr_(conv, r)` in `arch_util` (Aitken 2017, `weight[i::r*r] = base`), applied opt-in (`icnr=True` on the upsampler
  builder or one `init_icnr` call in `build_network` over `Conv2d -> PixelShuffle` pairs). Test: all sub-pixel phases equal
  at init, keys unchanged.

**A7 NAFNet** (new `neosr/archs/nafnet_arch.py`, `ARCHS` rows `nafnet` (width 32) / `nafnet_l` (64), test option; lane archs 2.3)
- ~150 lines from the paper / official MIT code: SimpleGate, simplified channel attention, LayerNorm2d as `F.layer_norm`
  NHWC (no custom autograd Function); enc `[2,2,4,8]`, mid 12, dec `[2,2,2,2]`; pad to `2^levels`; `scale` kwarg raises
  unless 1; official key names so NAFNet `.pth` load (test with a synthesised official-shaped state dict); export table entry.
- Cuttable: if the owner says 1x restoration is not a goal, skip A7 and the `psnr_loss` part of A4.

**A8 Hygiene** (`convert.py`, `neosr/runtime.py`, `pyproject.toml`; lane infra 3.7, 3.11)
- Constant folding on for float exports (verification already covers it); `-optimize` -> `ORT_ENABLE_BASIC` (portable) or
  label the EXTENDED file ORT-only; `fast_matmul` "medium" -> "high" with the comment fixed; `detect_anomaly` debug key
  (documented slow); drop `pytorch-optimizer` after grepping that nothing imports it.

**A9 Training guide** (docs; lane `competitor/recipes_evidence.md` sec. 1, infra 3.14; CONSENSUS Phase 1 #6)
- One short page: the L1-only edit of the GAN template; fidelity -> GAN with `pretrain_network_g` (fresh optimizer/EMA);
  x2 -> x4 with `strict_load_g = false`; lower LR for fine-tuning; which metric judges what (never the training loss);
  branching from an export. No second template set.

## Out of scope (do not start)
Phase B/C of `REVIEW.md` (OTF realism, R3GAN D, i-LN, KD/ECO, dynamo exporter, `wsd`, video, TOPIQ-NR, AESOP), template
default changes, GPU A/Bs, anything in the do-not-port list.

## Done means
All nine items committed one by one on `refactor/stabilize` with Codex-reviewed diffs, scoped tests green, the full suite
green once at the end, `stage3/PROGRESS.md` updated, and a final report: per item the commit, tests, Codex verdict and
open points; anything skipped named with the reason.
