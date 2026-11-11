"""One A/B arm: train a TOML through train.py's own machinery (load_config, validate_train,
setup_runtime, prepare_run, train) and log, beside the run, what scripts/ab/analyze.py reads:

    train.jsonl      every get_current_log() (the print_freq cadence): losses and diagnostics
    metrics.jsonl    every validation event: dataset means
    per_image.jsonl  every validation event: one row per image (id = path below the dataroot)
    done.json        only after the final validation of every val dataset: seconds, peak VRAM

    python scripts/ab/run_ab.py <repo> <config.toml> <out_dir> [--resume]

<repo> is the checkout whose code trains (a frozen copy keeps a running arm independent of later
edits); its train.py and neosr/ are imported from there. Seeded init and data order, fast kernels
(cudnn.benchmark, no strict determinism): arms differ by run-to-run noise, measured by a seed pair.
"""

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path


class Recorder:
    """The JSONL files and the per-image values of the validation event in progress."""

    def __init__(self, out: Path) -> None:
        self.t0 = time.time()
        self.files = {
            k: (out / f"{k}.jsonl").open("a", encoding="utf-8")
            for k in ("train", "metrics", "per_image")
        }
        self.ids: list[str] = []
        self.vals: list = []

    def write(self, key: str, row: dict) -> None:
        self.files[key].write(json.dumps(row) + "\n")
        self.files[key].flush()

    def clear(self) -> None:
        self.ids.clear()
        self.vals.clear()


def install_hooks(rec: Recorder) -> None:
    """Wrap the trainer's logging, validation and per-image metric points (their contracts are
    unchanged: every wrapper calls the original and returns its result)."""
    import torch  # ruff: ignore[import-outside-top-level]

    base = importlib.import_module("neosr.models.base")
    image = importlib.import_module("neosr.models.image")
    log_metrics, get_log = base.base.log_metrics, image.image.get_current_log
    validate, image_id, metric = (
        image.image.validate,
        image.image_id,
        image.image._metric,
    )

    def log_metrics_hook(self, dataset, results, current_iter, tb_logger):
        t = time.time() - rec.t0
        rec.write(
            "metrics", {"iter": current_iter, "t": t, "dataset": dataset, **results}
        )
        keys = list(self.metrics)
        rows = torch.stack(rec.vals).double().reshape(len(rec.ids), len(keys)).tolist()
        for rid, row in zip(rec.ids, rows, strict=True):
            row_d = dict(zip(keys, row, strict=True))
            rec.write(
                "per_image",
                {"iter": current_iter, "dataset": dataset, "id": rid, **row_d},
            )
        rec.clear()
        return log_metrics(self, dataset, results, current_iter, tb_logger)

    def image_id_hook(path, root):
        rel = image_id(path, root)
        rec.ids.append(str(rel))
        return rel

    def metric_hook(f, q, gt, warned):
        v = metric(f, q, gt, warned)
        rec.vals.append(v.detach().reshape(()).cpu())
        return v

    def get_log_hook(self):
        d = get_log(self)
        rec.write("train", {"iter": self.update, "t": time.time() - rec.t0, **d})
        return d

    def validate_hook(self, *args, **kwargs):  # per-image state is scoped to one event
        rec.clear()
        try:
            return validate(self, *args, **kwargs)
        finally:
            rec.clear()

    base.base.log_metrics = log_metrics_hook
    image.image.get_current_log = get_log_hook
    image.image.validate = validate_hook
    image.image._metric = staticmethod(metric_hook)
    image.image_id = image_id_hook


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="train one A/B arm with per-image logging")
    ap.add_argument("repo", type=Path, help="checkout whose code trains")
    ap.add_argument("config", type=Path, help="the arm's training TOML")
    ap.add_argument(
        "out", type=Path, help="run folder: JSONL logs, experiments/, done.json"
    )
    ap.add_argument(
        "--resume", action="store_true", help="auto-resume the arm's last state"
    )
    a = ap.parse_args(argv)
    repo, cfg_path, out = (p.resolve() for p in (a.repo, a.config, a.out))
    sys.path.insert(0, str(repo))
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    import torch  # ruff: ignore[import-outside-top-level]

    config = importlib.import_module("neosr.config")
    runtime = importlib.import_module("neosr.runtime")
    train = importlib.import_module("train")
    rec = Recorder(out)
    install_hooks(rec)
    cfg = config.load_config(
        str(cfg_path), is_train=True, root=out, debug=False, auto_resume=a.resume
    )
    config.validate_train(cfg)
    rt = runtime.setup_runtime(cfg)
    # seeded init and data order, fast kernels (manual_seed implies strict determinism in neosr)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(mode=False)
    torch.utils.deterministic.fill_uninitialized_memory = False
    train.prepare_run(cfg, ["run_ab.py", str(cfg_path)])
    train.train(cfg, rt)

    total = cfg["logger"]["total_iter"]
    with (out / "metrics.jsonl").open(encoding="utf-8") as f:
        final = {r["dataset"] for r in map(json.loads, f) if r["iter"] == total}
    wanted = {d["name"] for k, d in cfg["datasets"].items() if k != "train"}
    if final != wanted:  # no done.json: analyze.py and the queue see an incomplete arm
        print(
            f"incomplete: no final validation at {total} ({sorted(final)})",
            file=sys.stderr,
        )
        return 1
    done = {
        "seconds": time.time() - rec.t0,
        "update": total,
        "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    (out / "done.json").write_text(json.dumps(done), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
