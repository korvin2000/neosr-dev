# Stage 3: independent re-check of CONSENSUS.md (Fable 5.1, 2026-10-28)

Checked: `CONSENSUS.md`, the five lane reports (`competitor/*.md`), the traiNNer-redux-dev README, its full option
inventory (`traiNNer/utils/redux_options.py`, 969 lines of fields), and the `losses/ archs/ metrics/ data/ models/ scripts/`
listings of `/home/korvin/apps/traiNNer-redux-dev` (3a2d2549), against neosr `refactor/stabilize` (145024d): `LOSSES`,
`ARCHS`, `_metric_builders`, `SCHEDULERS`, `config.SCHEMA`, `checkpoint.py`, `convert.py`. Method: every C category was
diffed against a neosr counterpart, then each item was re-judged against the owner's rules (fast, stable, reliable, fully
functional, good code, no junk).

## Verdict
The consensus is right on both big calls: nothing in traiNNer-redux is worth copying as code, and the do-not-port list
stands. The inventory is complete: no discriminator, loss, metric, optimizer, scheduler, dataset option or trainer concept
in C is without a verdict. My corrections are about placement, not content: three items fell out of CONSENSUS.md
entirely, five were demoted too far, and three can share one mechanism. Total extra code stays small (every item S, two M).

## Inventory diff (C vs neosr)
| category | C has, neosr lacks | verdict |
|---|---|---|
| discriminators | R3GAN scalar (owner's code), MetaGAN2/3, MUNet, VGGStyle, multiscale PatchGAN | only R3GAN scalar (opt-in, with the rpgan+R1/R2 A/B); the rest skip (lane archs sec. 4: MetaGAN3 has no output head, GN cancels SN, no evidence) |
| losses | hinge, PSNR (log-MSE), per-loss schedule, D feature matching, AESOP, gradient-variance, HSLuv/Color/Luma, ConvNeXt/DINO perceptual (no gradient), contextual, FLIP, A-DISTS, HFEN/Sobel/Laplacian/TV, anime | hinge + PSNR + generic schedule (tiny, default-preserving); the rest skip or idea only |
| metrics | LPIPS, TOPIQ-NR, TOPIQ-FR PIPAL weights | LPIPS + PIPAL now; NR on request |
| optimizers | SOAP, ADOPT, StableAdamW, nadam | skip (no SR evidence; StableAdamW is damping) |
| schedulers | KneeLR, WSD, cosine-warmup, restart | nothing to copy; one closed-form `wsd` row only if E2 shows decay helps |
| data | AA/lanczos resize, independent stage probabilities + clean path, LQ-scale USM, paired+OTF mix (`dataroot_lq_prob`), GT-only dataset with plain resize, multi-root/weights, video datasets, real codecs | OTF realism track (goal-gated); codecs offline only; video on request |
| trainer | teacher/KD + ECO, `ema_switch_iter`, automations, VRAM manager, dynamic batch/patch, `watch`, `ignore_resume_networks` | KD/ECO goal-gated; everything automatic skip; "resume with a fresh D" is covered by export -> `pretrain_network_g` |
| export/IO | metadata in exports, safetensors r/w, dynamo exporter, onnxslim/BASIC, bf16 ONNX, JSON schema | metadata + safetensors read + folding/BASIC now; dynamo soon (torch 2.9.1 installed, TorchScript exporter deprecated); bf16 ONNX skip |

## Where this review differs from CONSENSUS.md
1. **Generic per-loss schedule** (`start_iter`/`ramp_iters`/`end_iter` on `Loss`): Phase 3 -> Phase A. It removes the
   GAN-only special case that already exists (`gan_loss.py:76-98`), makes E5 ("L1 first, then perceptual/GAN") a one-run
   config, keeps defaults, S.
2. **LPIPS**: "later" -> with TOPIQ-PIPAL. Same builder slot, ~60 lines; E4 (DISTS as a loss) has no second unbiased
   learned judge without it.
3. **Pretrained-load items** (lane infra #12) fell out of CONSENSUS: capped mismatch summary, old-layout ESRGAN key remap
   (fine-tuning OpenModelDB models is a core use), safetensors read. All in `checkpoint.py`/`esrgan_arch.py`, S -> Phase A.
4. **ONNX hygiene** fell out: constant folding is off for every float export and `-optimize` writes ORT-only EXTENDED graphs
   (`convert.py:449,466-470`). The 3-shape fp32+fp16 verification already guards it -> Phase A. The dynamo exporter moves
   to Phase B (forward compatibility, not quality).
5. **NAFNet**: Phase 2 -> last Phase A item, cuttable. "Fully functional" needs one real 1x restoration net (scale 1 is
   supported by the data path but no arch is a restoration net); add-only, official weights load, pairs with the 5-line
   PSNR loss. Drop both if 1x restoration is not a goal.
6. **i-LN**: Phase 3 -> Phase B (transformer goal). The only arch idea with paper evidence on four backbones neosr ships,
   and it addresses fp16 collapse that matters for neosr's fp16 ONNX path. Still opt-in, rewritten from the paper (C's
   version damps and is reload-inconsistent), gated by the tile-seam check and an A/B.
7. **R3GAN scalar D**: Phase 3 -> Phase B, bundled into the rpgan+R1/R2 A/B, which is incomplete without a norm-free D.
8. **ICNR**: stays P3, but it is a free extra arm of the realplksr A/B (init-only, opt-in).
9. **bench.py** also measures `compile_mode` variants (reduce-overhead, max-autotune): the step is launch-bound
   (`gpu_profile.md`), this is the cheapest speed lever left, and it is zero code beyond the script.

Everything else in CONSENSUS.md is confirmed: E0-E5 first and in that order; best-G export + metadata; scheduler `nets` +
resume semantics; TOPIQ-PIPAL; realplksr pre-LN opt-in; guide; drop `pytorch-optimizer`; `wsd` only after E2; OTF realism,
curation, KD/ECO goal-gated; the whole do-not-port list.

## Final list
**Phase 0 (owner, GPU, config only):** E0 evaluation protocol, E5 pretrain->GAN, E2 LR decay, E3 no-LDL, E4 DISTS, E1 optional.

**Phase A (code, default-preserving, CPU-verifiable; Opus session, `TASK_OPUS.md`):**
A1 exports + loading (metadata, best-G, safetensors read, load summary, ESRGAN remap) · A2 scheduler `nets` + resume +
range checks · A3 TOPIQ-PIPAL + LPIPS · A4 loss base schedule + hinge + psnr_loss · A5 `scripts/bench.py` · A6 realplksr
`layer_norm` + `icnr_` · A7 NAFNet · A8 hygiene (folding, BASIC, fast_matmul, detect_anomaly, drop pytorch-optimizer) ·
A9 training guide.

**Phase B (goal-gated code, opt-in, each with an A/B):** OTF realism (AA/lanczos resize first, independent probabilities +
clean path, LQ USM, offline `degrade.py`, paired+OTF mix) · R3GAN scalar D with the rpgan A/B · i-LN for HAT/SwinIR then
DRCT/SRFormer · KD/ECO via `Context.real` · weighted multi-dataset sampling + curation scorer · dynamo ONNX exporter ·
`wsd` row if E2 says decay helps · per-arch LR/patch table for archs in use.

**Phase C (on request):** TOPIQ-NR, D feature matching, AESOP, SCUNet, video (TSPANv2), JSON schema, safetensors export,
MS-SSIM reflect padding, real-time presets.

**Do not port:** as CONSENSUS.md (automations/damping of any name, the loss zoo, ParagonOTF and in-loop codecs, SOAP/ADOPT/
StableAdamW, KneeLR/restart, beaten archs, iLN/MetaGAN3/HyFreqMoE as written, registries, config-framework migration).

## Side finding
Both `neosr-dev` and `traiNNer-redux-dev` have a GitHub token in the remote URL (`git remote -v`). Revoke it and use a
credential helper.
