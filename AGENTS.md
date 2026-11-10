# Agent Guidelines

- Act as a senior PyTorch engineer and experienced python developer
- Use deep reasoning and show SOTA(state-of-the-art) results using: program of thought (PoT), chain of thought(CoT), Chain-Of-Reasoning (CoR)
- Also use additional techniques: Self-Refine, "priority hierarchy", intermediate summarization, "Chain Of Verification" (CoV) and other techniques for best results.
- Favor clear, readable, and compact implementations that follow clean code principles.
- Prefer well sturctured output like tables if writing documentation
- If unsure about a detail, make a best-effort assumption and clearly state it.

## Project facts (read before working)

- **What this project is:** neosr-dev is a reworked, optimised SISR trainer of its own. Its goals are clean and well-thought-out code, speed, a small footprint, good structure and compactness, and a core that is stable and reliable.
- **The goal is the framework, not the archs** (owner, 2026-11-10; `docs/refactor/DECISIONS.md`): the trainer and its ecosystem (losses, metrics, schedulers, optimizers, GAN implementations, utilities), its speed and stability, as a foundation for training *new* architectures. The existing archs rank near last (many are experimental or outdated and will never be trained): arch-specific fixes are deferred unless a framework contract needs them. Small, exact core fixes may be made without asking first.
- **traiNNer-redux is a competitor, not an upstream.** It is an alternative project whose code is often bloated (god classes, duplicated and dead paths). Use it only as a source of ideas and examples: take an idea when it is good, re-think and rewrite it to this project's standard, and never imitate its structure or copy its code. "traiNNer-redux has it" is not a reason to add something here.
- **No hasty decisions or conclusions.** On a disputed or open question (priorities, architecture, a default, how to read an ambiguous note in the docs) ask the owner instead of deciding; state what is verified and what is an assumption.
- **Status and open work:** `docs/refactor/STATUS.md` is the single status page; start there, then the stage log it points to. Update it when a track changes state. **What to do next, with everything needed to do it: `next_tasks.md`** (repository root); keep it current or delete what is done.
- **The owner sets the priorities** (`STATUS.md`, "Owner priorities"): a stable, reliable core first, then Studio, then new features. Do not start or promote a track because a document calls it a goal; when the next task is not named there, ask the owner which one before starting. Before writing something from scratch, check what upstream projects (traiNNer-redux by the-database) already have that can be adapted.
- **Two machines, two roles:**

  | machine | role |
  |---|---|
  | Linux workstation with an RTX 4090 (`/home/korvin/apps/neosr-dev`, venv with torch) | every real test and measurement: pytest (CPU and CUDA), training runs, A/Bs, benchmarks, profiling |
  | Windows workstation (`C:\work2000\neosr-dev`; Ryzen 5 5500U, no GPU; CPU-only `.venv` with torch) | code analysis, documentation, small core changes, the NeoSR Studio prototype (`studio/`) |

  The Windows CPU is slow: prefer tasks that need no compute (reading, design, docs, TypeScript), then tasks checked by a single test file or a `-k` selection (`.venv/Scripts/python.exe -m pytest tests/<file> -q -p no:cacheprovider`). Never run the full suite, per-arch or ONNX sweeps, benchmarks or training there; say what stays unverified and list core changes in `STATUS.md` ("Waiting for the Linux machine"). The venv is built by hand, not by `uv sync` (the lock pins CUDA wheels): recipe in `STATUS.md`. Torch-free checks: `python scripts/studio_surface.py --check`, `python scripts/studio_surface.py --validate <toml>`, and in `studio/`: `npm run typecheck`, `npx ng test --watch=false`, `npm run contract`.
- **The GitHub token in the `origin` remote URL is intentional** (private single-user machines; owner decision 2026-11-10). Do not report it as a finding and never print the remote URL (mask it in command output).
- **Trainer tables and Studio move together:** after changing `ARCHS`, `LOSSES`, `METRICS`, `OPTIMIZERS`, `SCHEDULERS`, `config.SCHEMA`, the gan types or `convert.py`'s export tables, run `python scripts/studio_surface.py`, commit `studio/shared/catalog/neosr-surface.json` and fix what the Studio contract test reports.
- **Evidence rules** (owner, `docs/refactor/stage2/DECISIONS.md`): stability comes from correctness, never from automatic damping; exact rewrites ship as defaults behind parity tests; objective or dynamics changes ship opt-in until an A/B on the owner's data decides.
