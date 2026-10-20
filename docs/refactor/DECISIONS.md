# Decisions

## Owner decisions (2026-10-20)
1. Delete legacy model types `default`, `simple_otf`, `eqr_otf`, `pancam_otf` and their exclusive datasets/archs
   (`pcpaired`, `pancamotf`, `pancamsingle`, `eqrotf`, `simple_otf` datasets, `otfgray` empty file, `pancam`/`eqrsrgan` archs).
   Only `image` and `otf` remain (used by all configs in `options/`).
2. Aether: register `aether_small/medium/large` AND make quantization-aware training (QAT) work end to end.
3. Git: branch `refactor/stabilize`, one commit per verified step. Never stage the owner's uncommitted files in the main
   checkout (`.codex/*`, `AGENTS.md` deletion, `info.txt`, `install.sh`, `options/train_esrgan2.toml`).
4. Old checkpoints: old `net_g/net_d *.pth` must load as pretrained weights; old `.state` resume is best effort;
   the new checkpoint format is the source of truth.
5. Loss pipeline requirement: every loss takes the data it needs from a shared context itself (and uses a model such as
   the discriminator when needed); no if-chains or per-loss `if/isinstance` branching in the training loop.

## Consensus decisions
See `analysis/consensus/*.md` (verdict tables) and `design/T*.md` (architecture). Summary of settled points is kept in
`PLAN.md`; disputed-then-resolved items are marked `majority` or `judge` in the consensus tables.

Notable refutations (do NOT "fix"):
- EMA `AveragedModel(use_buffers=False)` already copies live buffers (BN stats, spectral-norm u/v) on every update (F.1).
- `soap_sf.py` `parse_options` call lives inside a string literal and never runs (F.14).
- DropPath re-seeding does not clone weights in swinir/dat (they re-init after construction) but DOES in metagan (F.3).
