"""Per-term generator gradients of a training config, offline (stage 2 ST-01 / losses.md L-01).

    python scripts/term_grads.py -opt options/train_esrgan.toml [--net_g net_g.pth] [--net_d net_d.pth]
                                 [--batches 4] [--iter N]

Builds the model and the train loader as train.py does, then for a few training batches takes each generator
loss term's gradient over G's trainable parameters and prints, averaged over the batches, its norm, its share
of sum_t ||g_t||^2 and its cosine with the total gradient. A term with ~all of the share sets the update
direction alone; a term with ~0 share costs compute and does nothing. No optimizer steps, nothing is written
and the trainer is untouched: the batches go through the model's own optimize_parameters (same G forward,
clamp, autocast and Context) with G's backward replaced by this measurement and D's update left out.
"""

import argparse
import sys
from itertools import islice
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neosr.config import load_config, validate_train  # ruff: ignore[module-import-not-at-top-of-file]
from neosr.data import build_dataloader, build_dataset  # ruff: ignore[module-import-not-at-top-of-file]
from neosr.losses.base import Context  # ruff: ignore[module-import-not-at-top-of-file]
from neosr.models import build_model  # ruff: ignore[module-import-not-at-top-of-file]
from neosr.runtime import setup_runtime  # ruff: ignore[module-import-not-at-top-of-file]


def term_grads(model, batches, it: int | None = None) -> dict[str, dict[str, float]]:
    """{log key: {value, norm, share, cos}} of the G terms active at update `it` (default: the next
    update), averaged over `batches`. Parameters, optimizer state and clocks are left as they were
    (train-mode forwards still move buffers: BN statistics, spectral-norm vectors)."""
    phase, rows = model.g, []

    def measure(ctx: Context) -> None:
        # Loss.active (PLAN interface 1) where present: an inactive term is not part of this update
        terms = [t for t in phase.terms if getattr(t, "active", lambda _: True)(ctx.it)]
        with torch.autocast(**phase.autocast):
            values = [t.compute(ctx) for t in terms]
        scale = phase.scaler.get_scale() if phase.scaler.is_enabled() else 1.0  # fp16
        grads = []
        for v in values:
            gs = torch.autograd.grad(
                v * scale, phase.params, retain_graph=True, allow_unused=True
            )
            flat = [
                (torch.zeros_like(p) if g is None else g).flatten().float()
                for g, p in zip(gs, phase.params, strict=True)
            ]
            grads.append(torch.cat(flat) / scale)
        total = torch.stack(grads).sum(0)
        sq = torch.stack([g.square().sum() for g in grads])
        rows.append({
            t.log_key: {
                "value": float(v.detach()),
                "norm": float(s.sqrt()),
                "share": float(s / sq.sum()),
                "cos": float(F.cosine_similarity(g, total, dim=0)),
            }
            for t, v, g, s in zip(terms, values, grads, sq, strict=True)
        })

    saved = model.update, model.micro, model.d, phase.backward, phase.step
    target = model.update + 1 if it is None else it
    phase.backward, phase.step, model.d = measure, lambda: None, None
    try:
        for batch in batches:
            model.update, model.micro = target - 1, 0  # ctx.it = target on every batch
            model.feed_data(batch)
            model.optimize_parameters()
    finally:
        model.update, model.micro, model.d, phase.backward, phase.step = saved
    n = len(rows)  # the same terms on every batch: they depend on `it` only
    mean = {k: {s: sum(r[k][s] for r in rows) / n for s in rows[0][k]} for k in rows[0]}
    return dict(sorted(mean.items(), key=lambda kv: -kv[1]["share"]))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-opt", required=True, help="training config (.toml)")
    ap.add_argument(
        "--net_g", help="generator weights (default: the config's pretrain_network_g)"
    )
    ap.add_argument(
        "--net_d",
        help="discriminator weights (default: the config's pretrain_network_d)",
    )
    ap.add_argument(
        "--batches", type=int, default=4, help="training batches to average (default 4)"
    )
    ap.add_argument(
        "--iter",
        type=int,
        help="update index the terms see (start_iter, ramps, wavelet_init)",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.opt, is_train=True, root=ROOT)
    cfg["compile"] = False  # one-off measurement: compile time would dominate
    cfg.pop("val", None)  # no metric models
    p = cfg["path"]
    p.pop("resume_state", None)
    for net, path in (("g", args.net_g), ("d", args.net_d)):
        if path is not None:
            p[f"pretrain_network_{net}"] = str(Path(path).expanduser())
    validate_train(cfg)
    rt = setup_runtime(cfg)
    model = build_model(cfg, rt)
    ds_cfg = cfg["datasets"]["train"]
    loader = build_dataloader(build_dataset(ds_cfg), ds_cfg, runtime=rt)
    stats = term_grads(model, islice(loader, args.batches), args.iter)

    print(f"\n{'term':<16}{'value':>12}{'|g|':>12}{'share':>10}{'cos(g, total)':>15}")
    for key, s in stats.items():
        print(
            f"{key:<16}{s['value']:>12.4g}{s['norm']:>12.4g}{s['share']:>10.3%}{s['cos']:>15.3f}"
        )
    print(
        f"(G gradients over {args.batches} batch(es) of {ds_cfg['batch_size']}; share = ||g_t||^2 / sum)"
    )


if __name__ == "__main__":
    main()
