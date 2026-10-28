# Stage 3: what to take from traiNNer-redux (final, 3-model consensus)

Sources: five Opus research lanes (`competitor/*.md`), independent strategies from Fable 5.1 and Codex (GPT-6 Astra),
the Opus consensus draft (`competitor/CONSENSUS_DRAFT.md`) and both models' critiques of it. Disputes were settled by
checking code and evidence, not by vote; remaining differences are listed with the decision taken.

## Verdict (unanimous)
traiNNer-redux has almost nothing worth copying as code. Where the loops overlap neosr is ahead on correctness
(transactional skips, single loss weighting, feature losses that have gradients, verified export, metric nets built
once). Its value is as a map of gaps: LR decay for AdamW, self-describing exports and a best-model export, a faster
RealPLKSR variant, OTF realism for real-world models, and a better perceptual judge for validation. The next stage is
little code and many measured A/Bs; every code item keeps today's behaviour by default.

## Phase 0: evaluation first, then config-only experiments (GPU time, one job at a time, capped)
| # | What | Decision notes |
|---|---|---|
| E0 | Evaluation protocol: a larger fixed validation set (hundreds of held-out tiles), a second seed, one fixed judge that is never a training loss (TOPIQ-FR), one evaluation path, a fixed visual-check set | Fable and Codex both put this first: 67 tiles x 1 seed cannot resolve 0.05-0.1 dB |
| E5 | L1 pretraining then the GAN template vs the GAN template from scratch (`pretrain_network_g`; fresh optimizer/EMA) | highest prior (ESRGAN/Real-ESRGAN), zero code |
| E2 | LR decay vs constant (MultiStepLR on G only) | none of the 84 templates decays; AdamW runs at 2e-4 to 1M updates |
| E3 | template without LDL, judged by TOPIQ/visual on textured tiles (not PSNR) | LDL targets artifacts; its gradient share is ~1.4% of L1 |
| E4 | DISTS 0.15 as an addition and as a replacement for FDL, judged by a non-trained metric | only loss candidate with any on-data signal |
| E1 | matched esrgan run vs traiNNer's logged run | optional, last: cannot prove trainer parity either way |
Long-run acceptance must cross the templates' `wavelet_init = 80000` switch. Defaults change only after two seeds and
one longer run agree.

## Phase 1: small default-preserving code
1. Best-G export (`val.save_best`, exactly the evaluated EMA/SF weights, atomic) and reconstruction metadata in every
   export (`network_g`, scale, update, metric identity); `convert.py`/`test.py` read it, explicit CLI values win.
2. Scheduler: explicit `nets` targeting (decision: default = optimizers not in schedule-free mode, logged once; an
   explicit list may include them; Fable wanted SF exempt, Codex wanted no silent exemption). Resume: an unchanged
   scheduler config restores state exactly; a changed one is rebuilt and replayed to the saved step with a loud
   warning (Fable's replay, exact for composed warm-up/decay; Codex's concern answered by the warning), scheduler
   lists may change on resume (cooldown from any checkpoint); `validate_train` warns on `T_max < total_iter`.
3. Validation judges: TOPIQ-FR PIPAL weights as an option on the existing CFANet; templates name and enable the judge
   metric that is not a training loss; LPIPS later (registered lower-is-better).
4. `scripts/bench.py` from the stage-2 harness (steady state vs compile/startup, peak VRAM/host RAM).
5. RealPLKSR opt-in pre-LayerNorm (spandrel key names, eps 1e-6), measured with bench.py and an 8k A/B that also runs
   the GroupNorm-gamma init arm through the A/B harness (Codex: init changes dynamics; Fable: no knob unless it wins).
6. Training guide: the complete L1-only edit now; staged recipes after E5. No second template set.
7. Remove the unused `pytorch-optimizer` dependency.
8. A `wsd` scheduler row (one shape first) only if E2 shows decay helps.

## Phase 2: goal-gated tracks
OTF realism if real-world models matter (fixed synthetic val pairs via an offline `degrade.py` that reuses `Degrader`,
plus real LQ examples; torch antialiased resize modes; independent noise/JPEG/blur/clean probabilities with defaults
that reproduce today's draws); duplicate/leakage tooling (quality scoring only after inspection, no blanket removal of
flat tiles); per-arch LR/patch tuning for archs actually used (report sample budget and wall time); NAFNet for 1x
restoration; weighted multi-dataset sampling once a second source exists.

## Phase 3: research or on request
i-LN (transformer goal, tiled/deployment evaluation), R3GAN scalar D (after the metagan+rpgan+R1/R2 A/B), ICNR (if
checkerboards appear), teacher distillation / ECO (separately, via Context), hinge GAN, per-loss schedules, PSNR loss,
MS-SSIM padding, TOPIQ-NR, safetensors, exporter options, JSON schema, SCUNet / real-time presets / video.

## Do not port (unanimous)
Automatic damping under any name (DynamicLossScheduler, AGC, StableAdamW, adaptive_d, intelligent LR/early stop,
dynamic batch/patch, OOM auto-shrink, `ema_switch_iter`); the loss zoo (ConvNeXt/DINO perceptual have no gradient,
A-DISTS/MS-SSIM-L1/Luma buggy, 11 losses apply the weight twice); the ParagonOTF chain and in-loop codecs; SOAP/ADOPT,
KneeLR/restart schedulers; architectures beaten on traiNNer's own benchmark by archs neosr ships, iLN/MetaGAN3/
HyFreqMoE as written; registries and config-framework migrations.

## Risks
Weak test power (fixed by E0), metrics judging their own loss, no real-world validation data for OTF claims, reading
history as evidence, silent behaviour changes (keep flags and spandrel-compatible keys), template drift, compute/memory
(log ms/update, run jobs capped and one at a time).
