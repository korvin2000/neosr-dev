# Review brief: neosr-dev (super-resolution training framework)

Repo: /home/korvin/apps/neosr-dev  (Python 3.13, torch 2.9.1+cu128, RTX 4090; venv at .venv; run python as .venv/bin/python)
Competitor for comparison: /home/korvin/apps/traiNNer-redux-dev  (badly written overall, but some parts may be better/cleaner)

READ-ONLY: advise only, do not edit, create or delete any file in either repo. Do not commit.

## Goal
The project was neglected and was patched by many small automated "fix import error" commits. The owner wants it brought to a
reference-quality, stable, fast, compact, readable state. Your job is an independent, critical, constructive deep review that
will feed a refactoring plan. Be concrete: every finding cites `path:line` and explains the failure scenario.

## Scope (prioritised)
1. Core: train.py, test.py, convert.py, neosr/models/*, neosr/losses/*, neosr/data/*, neosr/utils/*, neosr/metrics/*, neosr/optimizers/*.
2. neosr/archs/*: only the shared helpers (arch_util.py, __init__.py), registration/import wiring, and anything the core depends on
   (e.g. discriminators metagan/unet/patchgan/ea2fpn/dunet/resgan, and vgg_arch used by losses). Do not review every SR architecture line by line.
3. Ignore .venv, uv.lock, common/ (except where imported), options/*.toml except to understand config keys actually in use.

## Questions to answer
A. STABILITY (highest priority). The current version has stability problems in real training. Find everything that can crash,
   silently stop, corrupt training dynamics, produce NaN/inf, break resume, or misbehave with: GAN + discriminator, gradient
   accumulation, AMP fp16/bf16 + GradScaler, EMA (AveragedModel), schedule-free optimizers, SAM (fsam), ECO, wavelet-guided,
   torch.compile, DDP/DataParallel, validation (tiled and non-tiled), checkpoint save/resume, the OTF degradation pipeline, dataloaders.
   Note: real logs exist in experiments/train_esrgan/*.log (config options/train_esrgan2.toml) - use them as evidence.
B. LOSS PIPELINE DESIGN. Today the training step computes losses through a long chain of `if self.cri_xxx:` blocks (and the
   competitor uses a loop over losses with `if label == ...` / isinstance branches inside the loop - also considered bad).
   The owner wants each loss class itself to take the data it needs from a shared context (sr output, gt, lq, upsampled lq,
   and models such as the discriminator / EMA generator when needed), so the training step is a single uniform loop with no
   per-loss branching. Propose the concrete minimal design (interfaces, where data lives, how GAN generator/discriminator
   losses fit, how config stays backward compatible with existing `*_opt` keys, logging keys), and list every current loss
   and how it maps onto it.
C. PERFORMANCE: per-iteration GPU syncs (.item(), isnan checks, printing), empty_cache/gc in hot paths, global
   torch.set_default_device("cuda"), dataloader/prefetch settings, channels_last, compile, redundant copies/clones, CPU<->GPU traffic.
D. STRUCTURE / READABILITY / COMPACTNESS: duplicated model hierarchies (models/default.py vs base.py/image.py/otf.py and
   simple_otf/eqr_otf/pancam_otf, image.py.helaman), dead/obsolete code and modules, sys.exit inside library code,
   duplicated save/load/retry logic, inconsistent config keys, swallowed exceptions. What can be deleted outright?
   What should be unified (three similar things -> one mechanism differing in data)?
E. COMPETITOR: what does traiNNer-redux implement better or more cleanly that is worth porting here (without fanaticism -
   only things with clear value for stability, speed or clarity)? Name the file there and the concrete idea.

## Reply format (keep it dense, <= ~5000 words)
1. Top 10 most important issues (one line each).
2. Findings table grouped by A/C/D: id, severity (critical/high/medium/low), `path:line`, defect, failure scenario, fix (1-2 lines), confidence.
3. Loss pipeline design proposal (B) with a short code sketch of the core (<= 80 lines) and the loss mapping table.
4. Competitor ideas worth porting (E), ranked by value/cost.
5. Deletion list (files/functions/options to remove) with justification.
6. Proposed refactor plan: ordered steps, which can run in parallel (disjoint files), and how to verify each.
7. Open questions / things you could not verify.
