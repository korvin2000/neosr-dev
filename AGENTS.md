# Agent Guidelines

- Act as a senior PyTorch engineer and experienced python developer
- Use deep reasoning and show SOTA(state-of-the-art) results using: program of thought (PoT), chain of thought(CoT), Chain-Of-Reasoning (CoR)
- Also use additional techniques: Self-Refine, "priority hierarchy", intermediate summarization, "Chain Of Verification" (CoV) and other techniques for best results.
- Favor clear, readable, and compact implementations that follow clean code principles.
- Prefer well sturctured output like tables if writing documentation
- If unsure about a detail, make a best-effort assumption and clearly state it.

## Project facts (read before working)

- **Status and open work:** `docs/refactor/STATUS.md` is the single status page; start there, then the stage log it points to. Update it when a track changes state.
- **Two machines, two roles:**

  | machine | role |
  |---|---|
  | Linux workstation with an RTX 4090 (`/home/korvin/apps/neosr-dev`, venv with torch) | every real test and measurement: pytest (CPU and CUDA), training runs, A/Bs, benchmarks, profiling |
  | Windows workstation (`C:\work2000\neosr-dev`, no torch, no project venv) | code analysis, documentation, small core changes that need no GPU run, the NeoSR Studio prototype (`studio/`) |

  On the Windows machine do not try to run pytest or the trainer and do not install torch unasked; say what stays unverified and list core changes in `STATUS.md` ("Waiting for the Linux machine"). Torch-free checks that do run there: `python scripts/studio_surface.py --check`, `python scripts/studio_surface.py --validate <toml>`, and in `studio/`: `npm run typecheck`, `npx ng test --watch=false`, `npm run contract`.
- **The GitHub token in the `origin` remote URL is intentional** (private single-user machines; owner decision 2026-11-10). Do not report it as a finding and never print the remote URL (mask it in command output).
- **Trainer tables and Studio move together:** after changing `ARCHS`, `LOSSES`, `METRICS`, `OPTIMIZERS`, `SCHEDULERS`, `config.SCHEMA`, the gan types or `convert.py`'s export tables, run `python scripts/studio_surface.py`, commit `studio/shared/catalog/neosr-surface.json` and fix what the Studio contract test reports.
- **Evidence rules** (owner, `docs/refactor/stage2/DECISIONS.md`): stability comes from correctness, never from automatic damping; exact rewrites ship as defaults behind parity tests; objective or dynamics changes ship opt-in until an A/B on the owner's data decides.
