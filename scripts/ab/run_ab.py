"""One A/B arm: train a TOML through train.py's own machinery (load_config, validate_train,
setup_runtime, prepare_run, train) and log, beside the run, what scripts/ab/analyze.py reads:

    train.jsonl      every get_current_log() (the print_freq cadence): losses and diagnostics,
                     mem_gib = the window's peak allocated VRAM (the [ mem ] of the log line)
    metrics.jsonl    every validation event: dataset means
    per_image.jsonl  every validation event: one row per image (id = path below the dataroot)
    done.json        only after the final validation of every val dataset: seconds, peak VRAM

    python scripts/ab/run_ab.py <repo> <config.toml> <out_dir> [--resume]

<repo> is the checkout whose code trains (a frozen copy keeps a running arm independent of later
edits); its train.py and neosr/ are imported from there. A fresh run empties the logs and done.json;
--resume keeps the events up to the resumed state and validates a final state whose validation was
cut off. Kernels follow the config (`deterministic`); arms differ by run-to-run noise, measured by a
seed pair.
"""

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

LOGS = ("train", "metrics", "per_image")


class Recorder:
    """The JSONL files and the per-image values of the validation event in progress."""

    def __init__(self, out: Path) -> None:
        self.t0 = time.time()
        self.files = {k: (out / f"{k}.jsonl").open("a", encoding="utf-8") for k in LOGS}
        self.ids: list[str] = []
        self.vals: list = []
        self.mem: float | None = None  # the current log window's peak, GiB

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
    trainer = importlib.import_module("train")
    peak = getattr(trainer, "window_peak_memory", None)  # older checkouts: no [ mem ]
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
        row = {"iter": self.update, "t": time.time() - rec.t0, **d}
        if rec.mem is not None:
            row["mem_gib"], rec.mem = rec.mem, None
        rec.write("train", row)
        return d

    # train.py reads the window's peak just before get_current_log (one log_vars literal)
    def peak_hook(device):
        m = peak(device)
        rec.mem = None if m is None else m[0]
        return m

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
    if peak is not None:
        trainer.window_peak_memory = peak_hook


def keep_until(out: Path, update: int | None, *, final: bool = False) -> None:
    """A fresh run (None) empties the logs; a resume keeps the events up to its state's update:
    later ones are re-logged by the resumed run (a validation at an update precedes its state).
    The final state is saved before its validation: `final` also drops that update's rows."""
    for key in LOGS:
        path = out / f"{key}.jsonl"
        if update is None or not path.exists():
            path.write_text("", encoding="utf-8")
            continue
        last = update - 1 if final and key != "train" else update
        with path.open(encoding="utf-8") as f:
            rows = [
                line for line in f if line.strip() and json.loads(line)["iter"] <= last
            ]
        path.write_text("".join(rows), encoding="utf-8")


def final_datasets(out: Path, total: int) -> set[str]:
    with (out / "metrics.jsonl").open(encoding="utf-8") as f:
        return {r["dataset"] for r in map(json.loads, f) if r["iter"] == total}


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
    (out / "done.json").unlink(missing_ok=True)  # written again only by a complete run
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    import torch  # ruff: ignore[import-outside-top-level]

    config = importlib.import_module("neosr.config")
    runtime = importlib.import_module("neosr.runtime")
    models = importlib.import_module("neosr.models")
    train = importlib.import_module("train")
    logger = importlib.import_module("neosr.utils.logger")
    cfg = config.load_config(
        str(cfg_path), is_train=True, root=out, debug=False, auto_resume=a.resume
    )
    config.validate_train(cfg)
    rt = runtime.setup_runtime(cfg)  # the config's `deterministic` decides the kernels
    train.prepare_run(cfg, ["run_ab.py", str(cfg_path)])
    state = cfg["path"]["resume_state"]  # <update>.state, or None for a fresh run
    resumed = None if state is None else int(Path(state).stem)
    total = config.total_iter(cfg)
    wanted = {d["name"] for k, d in cfg["datasets"].items() if k != "train"}
    if resumed == total and final_datasets(out, total) == wanted:
        print(f"complete: update {total} is validated", file=sys.stderr)
    else:
        # a final state whose validation was cut off: drop its partial rows, validate below
        keep_until(out, resumed, final=resumed == total)
    rec = Recorder(out)
    install_hooks(rec)
    train.train(cfg, rt)  # nothing to do when resumed at total_iter
    if resumed == total and final_datasets(out, total) != wanted:
        model = models.build_model(cfg, rt)  # the restored final weights
        save_img = (cfg.get("val") or {}).get("save_img", True)
        for vl in train.build_loaders(cfg, rt)[2]:
            model.validation(vl, total, None, save_img)

    if (
        final := final_datasets(out, total)
    ) != wanted:  # no done.json: an incomplete arm
        print(
            f"incomplete: no final validation at {total} ({sorted(final)})",
            file=sys.stderr,
        )
        return 1
    done = {
        "seconds": time.time() - rec.t0,
        "update": total,
        # a trainer that resets the peak per log window keeps the run's peak in run_peak_memory
        "peak_mem_gib": getattr(
            logger, "run_peak_memory", torch.cuda.max_memory_allocated
        )()
        / 2**30,
    }
    tmp = out / "done.json.tmp"
    tmp.write_text(json.dumps(done), encoding="utf-8")
    tmp.replace(out / "done.json")  # atomic: a reader sees no or a whole file
    return 0


if __name__ == "__main__":
    sys.exit(main())
