# Stage 3 consensus draft: what to port from traiNNer-redux into neosr

**Inputs:**
- The five lane reports in this directory: `archs`, `losses_metrics`, `data_degradation`, `training_infra` and `recipes_evidence` (**L**).
- The strategy from Fable 5.1 (**F**).
- The strategy from Codex GPT-6 Astra (**C**).

**Method.** I settled disagreements by checking the code and the evidence again, not by majority vote. Section 5 lists those checks. The work was read-only: no training runs, no probes, only file reads and greps.

**Legend:**
- **Cost:** S = a local change, about a day including tests; M = several interacting parts; L = a new pipeline.
- **Evidence:**
  - *strong*: measured on the owner's data, or a defect confirmed in code.
  - *medium*: a published result or a broad convention with a clear mechanism, not measured here.
  - *weak*: confounded logs, an inference-only benchmark, or reasoning alone.
- **Stances:** for / against / n.m. (not mentioned), with the source's own priority in brackets.

## 0. Verdict

- **traiNNer-redux has almost nothing neosr should copy as code.** Where the two training loops overlap, neosr is ahead on correctness:
  - exact resume and transactional skipped updates;
  - each loss weight applied once;
  - feature losses whose gradients reach the generator;
  - metric networks built once.

  What traiNNer does show is a few gaps in neosr:
  - the AdamW templates have no LR decay;
  - the best checkpoint is never exported, and exports do not record how the network was built;
  - a RealPLKSR variant that may be faster;
  - the OTF pipeline cannot produce some common LQ types (antialiased downscales, clean or JPEG-free samples).
- **Expected gains are modest on the owner's paired x4 data.** The two candidates most likely to help there are LR decay (fidelity at the end of a run) and pretraining with L1 before the GAN template. Both can be tested with configuration changes only (Phase 0). The OTF items matter only for real-world models.
- **Stage 3 should involve little code and many A/B tests.** Every code item below keeps current behaviour by default:
  - old `net_*.pth` files keep loading;
  - templates change only after an A/B;
  - lookup tables stay explicit, and losses read only from the loss Context.
- **All three sources agree on what not to port** (section 3): automatic damping, traiNNer's loss collection, the ParagonOTF chain, registries and the extra optimizers.
- **The lanes ranked five items too high:** NAFNet (P1), i-LN (P1), ConvNeXt loss (P2), R3GAN discriminator (P2), and curation as "where most of the remaining quality is". Both strategies lowered them, and the code checks support that.
- **Two lane claims are wrong** (section 2):
  - Gradient-norm logging is said to be missing. It already exists.
  - A traiNNer run collapse is blamed on a DINOv3 loss. That loss never produced a gradient.

## 1. Ranked plan

### Phase 0: experiments that decide what to build

These need GPU time only and no code beyond configs. Run them one at a time under `capped.sh`; together they take about a day.

| # | Experiment | Why | Evidence today | Stances L / F / C |
|---|---|---|---|---|
| E1 | **Same-architecture comparison with traiNNer.** (1) neosr `esrgan`, set up as close to traiNNer's 7k ESRGAN run as neosr allows: LQ 64, batch 8, AdamW 1e-4 with betas (0.9, 0.99), EMA 0.999 with `ema_power` 0.75, `chc_loss` set to plain Charbonnier, bf16, 8k updates. (2) neosr's GAN template on `esrgan`. Score both with `cross_eval.py` on val8. val67 was in traiNNer's training set. | Shows whether traiNNer's trainer or recipe has any advantage, before porting either. Read the result one way only: parity or better settles it; a deficit above 0.1 dB calls for a fully matched follow-up, not a port. | weak: today's gaps compare different architectures (L1: traiNNer +0.35-0.6 dB; GAN: neosr -0.8 dB PSNR but better DISTS and TOPIQ) | for (P1) / for (first) / for, with the caveat that old checkpoints cannot prove causation (2.8) |
| E2 | **LR decay vs constant LR.** realplksr, L1, 10k updates: constant vs `multisteplr` halvings at 5k/7.5k/9k (cosine as a third arm if cheap). Run the GAN arm only after item 4 is decided. | None of the 84 training templates decays the LR. This question needs no new code. | medium (convention, WSD papers); no SR number | for / for / for |
| E3 | **Template without LDL**, 8k. | If it ties, drop one term and one generator-side computation from 84 templates. | weak to medium. Probe: LDL's gradient is 1.4% of L1's at 28 dB. C: a global gradient norm understates a localised term, so the A/B decides. | for / for / for |
| E4 | **Template + `dists_loss` 0.15.** Judge by TOPIQ and PSNR (and LPIPS once item 1 lands), never by DISTS. Log ms/update. | The only loss candidate with any signal on the owner's data. | weak: one phase inside a traiNNer run, TOPIQ 0.707 → 0.745. Confounded with GAN weight 0.083 → 0.03 and 6k extra updates; that phase's ConvNeXt term had no gradient (2.1). | for (P2) / for / for (C adds an optional VGG-vs-FDL arm) |
| E5 | **L1 pretraining first.** realplksr L1 4k, then the template for 4k (`pretrain_network_g`), vs the template for 8k. | Decides what the training guide (item 8) recommends. | medium: ESRGAN and Real-ESRGAN papers. neosr: GAN from update 1 already beat a delayed GAN start (28.81 vs 28.69). | for / for / for |

**When to change a default.** Runs use one seed and 67 validation tiles, so a difference below about 0.05 dB is a tie (stage-2 rule). Change a template default only after two seeds and one longer run agree.

### Phase 1: small code changes that keep current behaviour by default (each S)

| # | Item: what and why | Approach | Cost | Evidence | Stances L / F / C |
|---|---|---|---|---|---|
| 1 | **Metrics for choosing checkpoints.** (a) Add TOPIQ-FR's PIPAL weights as `dataset = "kadid"` or `"pipal"` on the existing CFANet. The default stays kadid, and the cache key and best-value record include the dataset so the two scores never mix. (b) In the GAN templates, uncomment PSNR/SSIM/TOPIQ (and DISTS) in `[val]`, with a comment naming the metric that is not also a training loss. (c) Add an LPIPS metric (richzhang v0.1, alex by default, strict loading, the build-once / park-on-CPU path). Then score the stage-2 P/Q checkpoints with every metric, and choose the one that agrees with a visual check. | port and fix (PIPAL); config change; rewrite (LPIPS). Do not copy traiNNer's per-image rebuild or its `strict=False` loading. | S + S + S | medium. TOPIQ paper, Table II: rank correlation with human scores on PIPAL 0.81 for TOPIQ-PIPAL, 0.66 for DISTS, 0.60 for LPIPS. TOPIQ was trained on PIPAL there, so 0.81 is an upper bound. Code: pyiqa's two TOPIQ-FR configs differ only in the checkpoint. | for (P2) / for (do first) / for (rank 1; PIPAL first, LPIPS later): order settled in 2.7 |
| 2 | **Export the best checkpoint, and record the network config in exports.** (a) `val.save_best = "<metric>"` writes `net_g_best_<ds>_<metric>.pth` atomically after `validate()` returns, outside `eval_weights()`, with exactly the EMA/SF weights that were evaluated. (b) Exports become `{"params", "neosr": {"network_g", "scale", "update"}}`. `convert.py` and `test.py` read the `neosr` entry; command-line values win; old files fall back to the command line. Both neosr's loader and spandrel take `params` and ignore the extra key. | idea (a); port and fix (b) | S + S | strong (code). `convert.py:81-85` builds only `type` and `window_size`, so it cannot rebuild a network trained with other settings. Best values are tracked but never exported. | for (P2) / for (P1) / for (rank 1) |
| 3 | **Resuming with a changed scheduler config, plus a range check.** Today a resume silently keeps the saved milestones or `T_max` (`phase.py:265-266`, confirmed by a probe). New rule: if the scheduler config is unchanged, restore the full saved state, so resume stays bit-exact. If it changed, rebuild the scheduler from the config, restore only its step count, set each group's LR from the closed form at that count, and warn once per changed key. `LinearLR`, `MultiStepLR` and `CosineAnnealingLR` all have `_get_closed_form_lr`. Separately, `validate_train` rejects or warns when `T_max < total_iter`, because the LR then climbs back up. | idea | S | strong (probe and code) | for (P2, rule split by scheduler type) / for (P1, one rule) / for (keep exact resume and intentional changes separate): reconciled in 2.3 |
| 4 | **Which optimizers the scheduler applies to: decide first, then a few lines.** `[train.scheduler]` is built for every optimizer (`image.py:264-265`). A decay added to the templates would therefore also decay the discriminator's schedule-free `adan_sf`. Two options: a `nets` key (default all, today's behaviour), or schedule-free optimizers skip the scheduler through an attribute in the optimizer table. | idea | S | strong (code) | against (infra lane: "D follows the same schedule (the convention)") / n.m. / raises the confound |
| 5 | **A `"wsd"` scheduler row**, built only if E2 shows that decay helps. Warm-up-stable-decay (WSD): `lr = initial_lr · warm(t) · decay(t)`. The warm-up goes inside the row, because a prepended `LinearLR` loses it (probe); the row can also be written in chainable form. `decay_start` and `decay_iters` are given in updates or as fractions of `total_iter`; linear and 1-sqrt shapes. Copy nothing from traiNNer: its KneeLR gives the discriminator the generator's LR, and its WSD drops the warm-up. With item 3, a run can start its cooldown from any checkpoint. | rewrite from Hägele et al. 2024 | S (about 40 lines plus tests) | medium (convention and WSD papers); no SR number | for (P1) / for (rank 1) / for, but only once a cooldown branch is needed: settled in 2.2 |
| 6 | **RealPLKSR option `layer_norm = False`, plus the missing A-07 init knob.** (a) Pre-LayerNorm through `F.layer_norm` on the NHWC view; the post-branch GroupNorm becomes Identity. (b) An init setting for the GroupNorm gamma: `realplksr_arch.py` has none, although `DECISIONS.md:56` lists A-07 as kept opt-in. Keys do not change. Use key names `feats.N.layer_norm.{weight,bias}` and eps 1e-6, which is what spandrel 0.4.1 detects, so models trained with LN load in chaiNNer. First measure ms/update (eager and compiled, bf16), then run an 8k A/B: GroupNorm, GroupNorm with gamma 1e-2, LN. The architecture default stays GroupNorm. | port and fix. Do not copy traiNNer's hand-written LN (it overflows in fp16) or its `True` default (it breaks the official weights). | S | weak to medium. traiNNer's benchmark measures fp16 inference only (1.48-1.71x). The A-07 census shows a bad start (output std 2.6, generator gradient norm 29-49). Since stage 2, the `restore` clamp and clipping (close to scale-invariant under AdamW) have removed most of that early cost, so the case now rests mainly on training speed, which is still unmeasured. | for (P1) / for (P1) / for (rank 3): see 2.9 |
| 7 | **Benchmark script** `scripts/bench.py`, built from the stage-2 scratchpad harness. It reports median ms/update and peak memory for a given TOML, eager or compiled, bf16 or fp16. Never run it alongside training. | rewrite | S | Item 6 and any architecture decision need it | for (P3) / for (P3) / for (part of the test protocol): in Phase 1 because item 6 depends on it |
| 8 | **A one-page training guide.** Contents: the fidelity (L1-only) setup as a documented edit of a template; GAN from scratch vs L1 pretraining first, with neosr's own numbers (after E5); which metric judges which kind of run; how to start a cooldown (after item 5). No second set of templates. | docs | S | neosr's L1-only numbers already exist (R3: val67 29.91 dB at 5k) | for (P2, templates and guide) / for (P2, after E5) / for (publish L1-only recipes now): settled in 2.6 |

### Phase 2: tracks that wait for an owner goal or a Phase 0/1 result

| # | Item | Approach | Cost | Evidence | Stances L / F / C |
|---|---|---|---|---|---|
| 9 | **OTF track.** Only if real-world/OTF models matter; there are 42 `*_otf` templates. In order: (a) `scripts/degrade.py` runs a seeded `Degrader` over a GT folder and writes fixed OTF validation pairs and previews. It reuses `otf._load` instead of copying the pipeline, as traiNNer's 478-line script does. (b) Antialiased resizing: add torch's own `bicubic`/`bilinear` with `antialias=True` (torch 2.9.1 has CUDA kernels for both) to per-stage mode lists `resize_mode{,2,3}`. The defaults reproduce today's `_MODES` and random draws bit-for-bit. Write a custom resampler (lanczos, Mitchell) only if a target dataset needs it. (c) Add `blur_prob`, `noise_prob{,2}`, `jpeg_prob{,2}` and `clean_prob`, defaulting to 1/1/1/0 so today's behaviour is unchanged. (d) Later: sharpening halos at LQ scale (`sharpen_prob` defaults to 0) and an offline step with real codecs in `degrade.py`. | idea (a); port and fix (b, c) | S + S + S (+ S) | Code (strong): every sample gets noise in both stages, and JPEG in both stages when JPEG is on; the resize modes are not antialiased (`otf_degradation.py:36,102-128`). Probe: the owner's LQ matches antialiased cubic at 52.64 dB and lanczos2 at 52.71 dB, against 31.16 dB for neosr's current bicubic. The effect on quality is reasoning until an OTF A/B runs on the (a) set. | for (P1 for b and c; P2 for a) / for (P1, together with a) / for (Phase 2, torch's AA modes first): settled in 2.5 |
| 10 | **Dataset curation scorer and duplicate/leakage check**, writing `meta_info` lists. Columns: PSNR of a down-and-up round trip (how much SR detail a tile holds), Laplacian variance, a blockiness ratio computed in int32, noise σ. Then an A/B: full set vs the lowest 25% removed. Down-weight flat tiles; never delete them. | rewrite (traiNNer's blockiness metric wraps around in uint8, and its "aliasing" score removes detailed tiles) | M | weak on this data: the set is uniform, and the current train/validation split is already clean (probe p2) | for (P1) / for (P2) / for (Phase 2) |
| 11 | **Per-architecture LR and transformer patch size.** A/B tests: span/omnisr at LR 2e-4 vs 5e-4; dat at LQ patch 32 vs 64 with `accumulate` so both see the same number of samples. Caveat: the owner's GT tiles are 256 px, so at x4 an LQ patch of 64 is the whole tile and random cropping stops augmenting. HAT-L/DRCT-L at batch 8 and LQ 64 may not fit in 24 GB. | numbers only | S + GPU time | medium (paper tables); nothing measured on the owner's data | for (P2) / for (P2) / for (Phase 1 rank 2) |
| 12 | **NAFNet (1x restoration).** Only with a stated denoise/deblur goal, a 1x template and a 1x validation set. Port the official code (spandrel's copy is identical) with `F.layer_norm` in place of the custom autograd LN and spandrel-compatible keys. Check that official weights reproduce their outputs, and compare full-image with tiled inference (the network uses global pooling). | port and fix | S-M | strong for 1x restoration (paper); none for x4 SISR | for (P1) / for only with that goal (P2) / for only with that goal (Phase 2) |
| 13 | **Weighted sampling across several datasets** (paired and OTF mixed), once a second dataset or real-codec pairs exist. Port the idea only: traiNNer's version loads both samples for every index. | idea | M | reasoning | for (P2) / for (P3) / for (later) |

### Phase 3: research items or on request

| Item | Trigger or note | Stances L / F / C |
|---|---|---|
| i-LN (`norm="iln"`, rewritten from arXiv 2504.06629; HAT and SwinIR first) | Promote only if an fp16 evaluation of a neosr-trained HAT or SwinIR shows the problem the paper describes (feature magnitudes near 1e6, fp16 output collapse). The gain is +0.07 dB on full-size HAT (+0.6 dB in the lightweight setting). It normalises over the whole image, so tiled output differs from full-image output, and chaiNNer/vs-mlrt run tiled. | for (P1) / for (P2) / for (Phase 3) |
| R3GAN scalar discriminator (the owner's own code, port and fix) | Only after the pending stage-2 A/B of rpgan + R1/R2 on metagan. The owner's r3gan logs show no gain or instability, and a scalar discriminator gives only an image-level signal. | for (P2) / for (P3) / defer |
| ICNR initialisation | Run alongside an early-PSNR A/B. The `restore` clamp already fixed the early phase. | for (P2) / for (P3) / defer |
| Teacher distillation (`Context.real`) and ECO | If the owner distils compact models | for (P3) / for (P3) / defer |
| Hinge GAN, per-loss `start`/`ramp`/`end` for every loss, PSNR loss (restoration), MS-SSIM reflect padding, TOPIQ-NR (test sets without GT only) | On request | for (P3) / for (P3) / defer or n.m. |
| Reading safetensors; ONNX constant folding and ORT BASIC; the dynamo exporter; key remap for old ESRGAN files; JSON schema; config diff in the run log; peak VRAM in the log line; a template generator (once more than two defaults change); removing the unused `pytorch-optimizer` dependency (`pyproject.toml:28`, imported nowhere) | Maintenance and usability | for (P3) / for (P3) / n.m. |
| SCUNet, real-time presets, TSPANv2/video, RIB, presets with head_dim rounded up | Only with a matching goal (blind denoising, real-time, video, a future architecture stage) | for (P3) / against or n.m. / defer |

## 2. Disagreements and how they were settled

**2.1 ConvNeXt perceptual loss (recipes lane P2): do not port.**
- traiNNer's `_extract_feats` is decorated `@torch.no_grad()` and also computes the prediction's features (`convnext_perceptual_loss.py:118,164`). The file has a single commit (2025-11-08), its working copy is unchanged, and the 43k phase's config, `ESRGAN7.yml` (dated 2026-01-07), uses it. So that term had no gradient.
- The DINO loss is the same: `dino_perceptual_loss.py:90`, last changed 2025-11-16, while Exp3B ran on 2026-01-14.
- This also refutes the recipes lane's verdict 3. The "DINOv3 term at 95% of the loss" in Exp3B carried no gradient and cannot explain that collapse. Other candidates are `moe_router_noise: 0.07` and the r3gan and discriminator settings.
- neosr's small FDL weight is justified by its own A/B, not by that log. F and C agree.

**2.2 WSD now (L, F) or later (C).**
- Whether decay helps at all can be tested without code (E2, using `multisteplr`).
- Over `multisteplr`, the WSD row only adds its shape and the ability to start a cooldown from any checkpoint. That ability fits how the owner works: runs stop at arbitrary points, and traiNNer phases were made by editing the config and resuming.
- Settled: E2 first; build the row if E2 is positive. Build item 3 either way.
- Remaining difference: F would build the row now because it is small. Either order costs little.

**2.3 Resume semantics.** L splits the rule by scheduler type, F wants one rule where the config wins, and C wants exact resume kept separate from intentional changes.
- In the code, optimizer settings come from the saved state, with a loud warning per changed key (`phase.py:255-262`). The saved scheduler state silently replaces the config (`phase.py:265-266`).
- Recomputing torch's recursive cosine schedule in closed form matches only to four digits (infra probe), so always recomputing would break bit-exact resume.
- Item 3's rule meets all three positions: an unchanged config resumes exactly; a changed config takes effect from the restored step count, with a warning.
- Correction to F: the range check is `T_max < total_iter` (the LR re-warms), not `T_max <= total_iter`.

**2.4 Gradient-norm logging (recipes lane P2, F P1 #7) already exists.**
- `image.py:359-361` adds `grad_norm_g/d` and `clipped_*` to the meter, and the periodic log line prints it (`train.py:239`).
- Only peak VRAM is missing (P3). C is right.

**2.5 Resampler: custom kernel table (data lane, F) or torch's antialiased modes (C).**
- The data lane's own probe puts torch's antialiased bicubic within 0.07 dB of lanczos2 for the owner's LQ.
- torch 2.9.1 ships CUDA kernels for antialiased bicubic and bilinear.
- torch's modes therefore close the measured gap in a few lines. A custom resampler waits until some dataset needs other kernels.
- On phase placement, all three describe the same item under the same condition: an OTF goal plus a fixed OTF validation set. It sits in Phase 2 because nothing in it can be judged before 9a exists.

**2.6 Fidelity templates now (C), after E5 (F), or templates plus a guide (L).**
- Adding 84 more hand-maintained templates would repeat the drift of stage 2's edit of all 84.
- The L1-only setup is a single edit of a template, and its numbers already exist.
- Settled: document it now (item 8). Recommend L1 pretraining first only if E5 supports it. Add fidelity templates only together with a template generator.

**2.7 Is LPIPS a prerequisite (F) or can it follow (C)?**
- neosr already has TOPIQ-FR (KADID weights), and no recipe uses it as a training loss, so E4 can be judged today.
- LPIPS adds a second independent judge and numbers comparable with papers. It stays in Phase 1, after PIPAL.

**2.8 Value of E1.**
- C: historical checkpoints cannot prove trainer parity. That is true for causation.
- E1 is therefore set up as a matched run and read in one direction only, and it costs two runs.
- With that setup, all three agree.

**2.9 RealPLKSR LN priority.** All three keep it opt-in and measured. The checks added three facts:
- The A-07 GroupNorm-gamma init that the lanes want to compare against is not in the code.
- The init argument is weaker since stage 2: the `restore` clamp gained 4.2 dB at 500 updates, and clipping is close to scale-invariant under AdamW (stage-2 decision 5).
- chaiNNer compatibility through spandrel requires the `layer_norm` key name and eps 1e-6.

**2.10 NAFNet, i-LN, R3GAN discriminator, ICNR and curation (lanes P1/P2).** Both strategies moved these to conditional, P2 or P3. The evidence each lane cites supports the move:
- NAFNet's value is entirely in 1x restoration, and neosr has no template or validation set for that.
- i-LN's headline gain is in the lightweight setting, and it changes tiled outputs.
- The R3GAN paper's evidence is image generation, and the owner's r3gan logs are negative.
- ICNR only affects initialisation.
- Curation has no measured number on this data.

**Still open (the owner decides):**
- Item 4: which optimizers the scheduler drives when the discriminator is schedule-free.
- Whether OTF/real-world models (item 9) or 1x restoration (item 12) is a goal now. That would move those tracks ahead of Phase 1 items 6-8.
- i-LN as P2 or P3.

## 3. Do not port

| What | Why |
|---|---|
| DynamicLossScheduler, AGC (adaptive gradient clipping), adaptive_d, intelligent LR / early stop, dynamic batch and patch size, OOM auto-shrink, zero-config, dataset analyzer, `ema_switch_iter` | Owner rule (automatic damping). All 74 of the owner's logged traiNNer launches had automations off, so there is not even anecdotal evidence for them. |
| ParagonOTF physical degradation chain | Lens distortion transposes the LQ even at strength 0 (probe). Motion blur and rolling shutter misalign LQ and GT. Exposure and colour shifts cannot be identified from a patch. Effects applied at GT scale are averaged away by the downscale. Chromatic aberration does nothing. About 20 keys in its showcase config are silently ignored. |
| PIL/ffmpeg codecs inside the training loop | About 150 ms per AVIF batch, run synchronously; values truncated, not rounded; one quality per batch; on error the input is silently returned. Real codecs belong offline (item 9d). |
| traiNNer losses: ConvNeXt/DINO perceptual, A-DISTS, MS-SSIM-L1, Luma, HSLuv, HFEN/Sobel/Laplacian/TV/checkerboard, contextual, FLIP, anime, CLIP, feature-matching wiring | ConvNeXt and DINO have no gradient. A-DISTS, MS-SSIM-L1 and Luma are buggy. The rest duplicate neosr terms or are off-task. 11 traiNNer losses apply `loss_weight` twice, so its recommended weights are not evidence. |
| SOAP, ADOPT, StableAdamW; KneeLR, CosineAnnealingRestart, ReduceLROnPlateau, the Sequential/Chained rows, `restart_scheduler` | No SR evidence. SOAP needs 2-3x Adam's memory, and StableAdamW's clipping is damping. KneeLR gives the discriminator the generator's LR. CosineAnnealingRestart raises an error past its periods, and ReduceLROnPlateau at its first step. One `wsd` row (item 5) covers the need. |
| GRL, Swin2SR, HiT-SIR/SNG, EMT, ELAN, SeemoRe, DWT, SRFormerV2, FDAT/TFDAT, community and bespoke networks, DAT/HAT-iLN as written, MetaGAN2/3, MUNet, HyFreqMoE as written | On traiNNer's own benchmark they are beaten by architectures neosr already ships, have no controlled numbers, or are broken. MetaGAN3 has no output head. traiNNer's iLN is EMA damping with a buffer that is not saved. HyFreqMoE injects its auxiliary loss as a gradient hook, and its step counter is lost on resume. |
| spandrel/timm registries, msgspec config, `arch_info.py` | They break the explicit-table rule, and neosr's schema already validates the config. |
| bf16 ONNX, blended tiling, crash checkpointing, validation that silently skips images that fail, video datasets | Worse than what neosr does now, or out of scope. |

## 4. Main risks

1. **Weak test power.** One seed, 67 tiles and 8-10k updates: anything below 0.05 dB is a tie, and the effects of schedules and patch size show only over long runs. Require two seeds and one longer run before changing a default. Otherwise the plan only produces opt-in flags that nobody turns on.
2. **Metrics judging their own loss.** Once DISTS (or FDL-like features) is a training loss, DISTS no longer judges fairly. Choose checkpoints with a metric that is never trained on, name it in the templates, and keep visual checks in the protocol.
3. **No real-world validation data.** Every OTF and curation item is reasoning until a fixed OTF validation set (9a) or real LQ images exist. No-reference metrics reward hallucinated detail.
4. **Reading history as evidence.** traiNNer's phases, logged loss values and inference benchmarks are not controlled evidence. The ConvNeXt/DINO case shows a loss value appearing in the log while the term produced no gradient.
5. **Silent behaviour changes.** Normalisation variants, scheduler edits and tiled inference can change outputs while checkpoints still load. Keep new behaviour behind explicit flags, keep spandrel-compatible keys, and compare tiled with full-image output.
6. **Template drift.** Each new option (`wsd`, `layer_norm`, `clean_prob`, patch size) multiplies variants across 84 templates. Decide defaults in batches, and build the template generator if more than two change.
7. **Compute and memory.** Larger patches and extra perceptual networks can cancel out generator speed-ups, so log ms/update in every A/B run. The machine killed the desktop through out-of-memory twice. Run jobs one at a time under `capped.sh`, one GPU job at a time.

## 5. Checks made for this draft (read-only)

**neosr: templates and training loop**
- No `[train.scheduler]` in any of the 84 training templates (127 option files in total).
- 26 templates use LQ `patch_size = 32` (13 transformer architectures, paired and OTF); 58 use 64. `patch_size` is the LQ size (`paired_dataset.py:40`).
- `models/base.py:40-43,65-71`: two scheduler rows, built once per optimizer.
- `image.py:255-265`: the prepended `LinearLR`, and a scheduler for every optimizer.
- `phase.py:255-266`: optimizer-setting warnings, and the silent scheduler restore.
- `image.py:359-361` and `train.py:239`: gradient-norm logging.

**neosr: OTF pipeline**
- `otf_degradation.py:36,102-128,152-176`: resize modes without antialiasing; noise in every stage; JPEG in every stage when enabled.
- torch 2.9.1 ships the `_upsample_bicubic2d_aa` and `_upsample_bilinear2d_aa` CUDA kernels.

**RealPLKSR and exports**
- `realplksr_arch.py:83-96,104-117`: GroupNorm gamma initialised to 1; no init setting and no LN option.
- spandrel 0.4.1, `PLKSR/__init__.py:123` and `RealPLKSR.py:11-22`: key name `layer_norm`, eps 1e-6.
- spandrel `canonicalize.py:25-36` and neosr `checkpoint.py:80-82`: both take the `params` entry.

**Loss Context, convert.py and TOPIQ**
- `losses/base.py:33-58`: the fields the loss Context exposes.
- `convert.py:81-85`.
- `metrics/topiq.py:268` against pyiqa `default_model_configs.py:404-421`.

**traiNNer-redux**
- `convnext_perceptual_loss.py:118,164`: one commit (2025-11-08), working copy unchanged; `experiments/4x_ESRGAN/ESRGAN7.yml:135` (file dated 2026-01-07).
- `dino_perceptual_loss.py:90`: last commit 2025-11-16; `experiments/Experimental3B/new_arch2.yml` (2026-01-14) sets `moe_router_noise: 0.07`.

**Dependency**
- `pyproject.toml:28` declares `pytorch-optimizer`; nothing in `neosr/`, `tests/` or the entry points imports it.

**Security note** (raised by two lanes and by F): the `origin` URL in `/home/korvin/apps/traiNNer-redux-dev/.git/config` contains a GitHub token in plain text. Rotate the token and switch to a credential helper.
