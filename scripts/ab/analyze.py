"""A/B verdicts from scripts/ab/run_ab.py runs: dataset means, paired per-image differences against a
reference arm with a 95 % bootstrap CI over source groups, the gate (README.md) and training
diagnostics.

    python scripts/ab/analyze.py --runs RUNS --data DATA REF ARM [ARM ...]
        [--seed-ref REF2] [--iter 8000] [--dataset dev] [--independent]
        [--window FROM:TO --null X/Y,...] [--report-only METRIC]

REF may pool runs of one config, A+B+...: their per-image mean. --window is the D10 gate
(docs/refactor/DECISIONS.md): every metric is the mean over the validated updates in [FROM, TO],
and the threshold tau = max(3 x the RMS of the same-config pairs' deltas, the floor below).

RUNS holds one folder per arm (<arm>/per_image.jsonl, metrics.jsonl, train.jsonl, done.json).
DATA is a folder (sources.json, optional groups_eval.json and exclude.txt, <dataset>_gt/) or a
JSON file {"sources": ..., "groups": ..., "exclude": ..., "manifest": {"dev": "<gt dir>", ...}}
with paths relative to it. sources.json: a list of groups of GT file names (one group = one source),
required unless --independent (every image its own group); groups_eval.json: {image id: group}
links added on top; exclude.txt: ids left out of every comparison; the manifest: the GT folder
whose images every arm must have scored exactly once (ids as the trainer's: the path below the
folder without suffix). Every missing input the gate needs exits instead of weakening it.
"""

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SIGN = {"psnr": 1, "topiq": 1, "lpips": -1, "dists": -1}  # +1: higher is better
PERCEPTUAL = (
    "topiq",
    "lpips",
    "dists",
)  # the verdict's metrics; PSNR is reported (fidelity)
ABS_FLOOR = {"psnr": 0.05}  # dB
REL_FLOOR = 0.01  # of the reference mean, for the perceptual metrics
NULL_K = 3.0  # D10: tau = max(NULL_K x null RMS, floor)
RMS_ONLY = ("topiq",)  # D10: no 1 % term (it is ~10x TOPIQ's run-to-run noise)
DIAG = (
    "out_d_margin",
    "r1",
    "r2",
    "l_d_reg",
    "grad_norm_d",
    "clipped_d",
    "l_g_gan",
    "grad_norm_g",
)


@dataclass
class Data:
    # image id -> source group; None: independent images
    group: dict[str, int | str] | None
    manifest: dict[str, set[str]]  # dataset -> the ids every arm must have scored
    exclude: set[str]


def _ids(folder: Path) -> set[str]:
    """The trainer's image ids (data_util.image_id): the path below the folder without suffix."""
    return {
        str(p.relative_to(folder).with_suffix(""))
        for p in folder.rglob("*")
        if p.is_file() and not p.name.startswith(".")
    }


def load_data(path: Path, *, independent: bool = False) -> Data:
    if path.is_dir():
        spec = {
            "sources": "sources.json",
            "groups": "groups_eval.json",
            "exclude": "exclude.txt",
            "manifest": {p.name[:-3]: p.name for p in path.glob("*_gt") if p.is_dir()},
        }
        root = path
    else:
        spec, root = json.loads(path.read_text(encoding="utf-8")), path.parent

    def opt(key: str) -> Path | None:
        """The file a key names; None if unnamed. A JSON spec's named file must exist."""
        if not spec.get(key):
            return None
        p = root / spec[key]
        if not p.exists():
            if path.is_dir():
                return None
            sys.exit(f"{path}: {key} file {p} is missing")
        return p

    group: dict[str, int | str] | None = None
    if not independent:
        if (src := opt("sources")) is None:
            sys.exit(
                f"{path}: no sources file (source groups); --independent goes without"
            )
        groups = json.loads(src.read_text(encoding="utf-8"))
        group = {
            str(Path(n).with_suffix("")): i for i, g in enumerate(groups) for n in g
        }
        if links := opt("groups"):
            group |= json.loads(links.read_text(encoding="utf-8"))
    exclude = (
        set(ex.read_text(encoding="utf-8").split()) if (ex := opt("exclude")) else set()
    )
    manifest = {ds: _ids(root / d) for ds, d in spec.get("manifest", {}).items()}
    return Data(group, manifest, exclude)


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def per_image(runs: Path, data: Data, arm: str, it: int, ds: str) -> dict[str, dict]:
    """{id: row} of `arm` at update `it` on `ds`, excluded ids dropped; {} if not evaluated. Exits
    on duplicate or missing ids (against the manifest) and on a non-finite metric."""
    rows = [
        r
        for r in jsonl(runs / arm / "per_image.jsonl")
        if r["iter"] == it and r["dataset"] == ds
    ]
    if not rows:
        return {}
    ids = [r["id"] for r in rows]
    if (want := data.manifest.get(ds)) is None:
        sys.exit(f"{ds}: no manifest (<dataset>_gt folder) to check the arms against")
    if len(ids) != len(set(ids)) or set(ids) != want:
        sys.exit(
            f"{arm} {ds} @{it}: {len(ids)} rows, {len(set(ids))} unique, manifest {len(want)}"
        )
    if missing := [m for m in SIGN if m not in rows[0]]:
        sys.exit(f"{arm} {ds} @{it}: gate incomplete, no {', '.join(missing)}")
    if not all(math.isfinite(r[m]) for r in rows for m in SIGN if m in r):
        sys.exit(f"{arm} {ds} @{it}: non-finite metric")
    return {r["id"]: r for r in rows if r["id"] not in data.exclude}


def mean_rows(parts: list[dict[str, dict]]) -> dict[str, dict]:
    """{id: {metric: mean over the parts}}: the parts are one arm's updates or pooled runs."""
    if any(p.keys() != parts[0].keys() for p in parts):
        sys.exit("mean_rows: the parts scored different image sets")
    return {
        i: {m: float(np.mean([p[i][m] for p in parts])) for m in SIGN} for i in parts[0]
    }


def window_iters(runs: Path, names: list[str], ds: str, span: str) -> list[int]:
    """The updates in FROM:TO at which any of the named runs was validated on `ds`: a run that
    lacks one is not evaluated (an arm) or stops the gate (a reference or null run), so a run cut
    short cannot shorten the window."""
    try:
        lo, hi = (int(v) for v in span.split(":"))
    except ValueError:
        sys.exit(f"--window {span}: FROM:TO, two updates")
    its = sorted({
        r["iter"]
        for name in names
        for r in jsonl(runs / name / "per_image.jsonl")
        if r["dataset"] == ds and lo <= r["iter"] <= hi
    })
    if not its:
        sys.exit(f"no {ds} validation in {span}")
    return its


def null_rms(rows, pairs: list[list[str]]) -> dict[str, float]:
    """Per metric: the RMS over same-config run pairs of their mean paired difference."""
    d: dict[str, list[float]] = {m: [] for m in SIGN}
    for x, y in pairs:
        rx, ry = rows(x), rows(y)
        if not rx or rx.keys() != ry.keys():
            sys.exit(f"--null {x}/{y}: not both evaluated on the same images")
        for m in d:
            d[m].append(float(np.mean([rx[i][m] - ry[i][m] for i in rx])))
    return {m: float(np.sqrt(np.mean(np.square(v)))) for m, v in d.items()}


def taus(ref: dict, rms: dict[str, float]) -> dict[str, float]:
    """D10's threshold per metric: max(NULL_K x null RMS, 0.05 dB PSNR or 1 % of the reference
    mean), without the 1 % term for RMS_ONLY."""
    out = {}
    for m in SIGN:
        rel = (
            0.0
            if m in RMS_ONLY
            else REL_FLOOR * abs(np.mean([r[m] for r in ref.values()]))
        )
        out[m] = max(NULL_K * rms[m], ABS_FLOOR.get(m, float(rel)))
    return out


def bootstrap(
    diff: np.ndarray, groups: np.ndarray, n_boot: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of the per-image differences and its 95 % CI: groups drawn with replacement, the
    resample's mean = sum of its groups' differences / their image count (cluster bootstrap)."""
    _, inv = np.unique(groups, return_inverse=True)
    sums, cnt = np.bincount(inv, diff), np.bincount(inv)
    pick = np.random.default_rng(seed).integers(0, len(cnt), (n_boot, len(cnt)))
    boot = sums[pick].sum(1) / cnt[pick].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def paired(a: dict, b: dict, data: Data, metrics=tuple(SIGN)) -> dict:
    """arm a - reference b per metric: (mean, CI low, CI high), plus n and the group count."""
    if set(a) != set(b):
        sys.exit("paired: the arms scored different image sets")
    ids = sorted(a)
    if data.group is None:
        groups = np.array(ids)
    elif missing := [i for i in ids if i not in data.group]:
        sys.exit(
            f"paired: {len(missing)} images without a source group, e.g. {missing[0]}"
        )
    else:
        groups = np.array([str(data.group[i]) for i in ids])
    if len(set(groups)) < 2:
        sys.exit(f"paired: {len(set(groups))} source group(s), a CI needs at least 2")
    out: dict = {"n": len(ids), "groups": len(set(groups))}
    for m in metrics:
        out[m] = bootstrap(np.array([a[i][m] - b[i][m] for i in ids]), groups)
    return out


def floors(ref: dict, seed_ref: dict | None) -> dict[str, float]:
    """The gate's floor per metric: max(seed spread, 0.05 dB PSNR, 1 % of the reference mean)."""
    out = {}
    for m in SIGN:
        if m not in next(iter(ref.values())):
            continue
        mean = float(np.mean([r[m] for r in ref.values()]))
        spread = 0.0
        if seed_ref:
            spread = abs(mean - float(np.mean([r[m] for r in seed_ref.values()])))
        out[m] = max(spread, ABS_FLOOR.get(m, REL_FLOOR * abs(mean)))
    return out


def status(m: str, mean: float, lo: float, hi: float, floor: float) -> str:
    """better / worse beyond the floor with a CI that excludes 0 in that direction, else same."""
    s = SIGN[m]
    if s * mean > floor and (lo > 0 if s > 0 else hi < 0):
        return "better"
    if s * mean < -floor and (hi < 0 if s > 0 else lo > 0):
        return "worse"
    return "same"


def verdict(states: dict[str, str], gated=PERCEPTUAL) -> str:
    """LOSS if a gated perceptual metric is worse; WIN if >= 2 of them (all, when fewer are gated)
    are better and none worse; else TIE."""
    perc = [states[m] for m in gated if m in states]
    if "worse" in perc:
        return "LOSS"
    return "WIN" if perc and perc.count("better") >= min(2, len(perc)) else "TIE"


def diagnostics(runs: Path, arm: str) -> dict:
    rows = jsonl(runs / arm / "train.jsonl")
    out: dict = {}
    for key in DIAG:
        v = np.array([r[key] for r in rows if key in r], dtype=float)
        if v.size:
            q = len(v) // 4 or 1
            out[key] = {
                "first": v[:q].mean(),
                "last": v[-q:].mean(),
                "min": v.min(),
                "max": v.max(),
            }
    out["skips"] = sum(int(r[k]) for r in rows for k in r if k.startswith("skipped_"))
    t = [(r["iter"], r["t"]) for r in rows if "t" in r]
    # median over log windows: validation pauses and a resume's restarted clock drop out
    if steps := [
        (t1 - t0) / (i1 - i0)
        for (i0, t0), (i1, t1) in itertools.pairwise(t)
        if i1 > i0 and t1 > t0
    ]:
        out["ms_update"] = 1000 * float(np.median(steps))
    if mem := [r["mem_gib"] for r in rows if "mem_gib" in r]:  # window peaks
        out["mem_gib"] = {
            "first": mem[0],
            "steady": float(np.median(mem)),
            "max": max(mem),
        }
    if (done := runs / arm / "done.json").exists():
        rec = json.loads(done.read_text(encoding="utf-8"))
        # runs finished before steady_mem_gib name the same run peak peak_mem_gib
        if "peak_mem_gib" in rec:
            rec["run_peak_gib"] = rec.pop("peak_mem_gib")
        out |= rec
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="A/B verdicts with a CI over source groups"
    )
    ap.add_argument("ref", help="reference arm; A+B+... pools runs of one config")
    ap.add_argument("arms", nargs="+", help="arms compared with the reference")
    ap.add_argument(
        "--runs", type=Path, required=True, help="folder of arm run folders"
    )
    ap.add_argument("--data", type=Path, required=True, help="data folder or JSON spec")
    ap.add_argument(
        "--seed-ref", help="the reference's second seed (the floor's seed spread)"
    )
    ap.add_argument(
        "--iter", type=int, default=8000, help="update compared (default 8000)"
    )
    ap.add_argument(
        "--window",
        metavar="FROM:TO",
        help="D10: mean over the validated updates in [FROM, TO] instead of --iter",
    )
    ap.add_argument(
        "--null",
        metavar="X/Y,...",
        help="same-config run pairs (with --window): tau = max(3 x their RMS, floor)",
    )
    ap.add_argument(
        "--report-only",
        action="append",
        default=[],
        choices=PERCEPTUAL,
        metavar="METRIC",
        help="a metric the arms train on: reported, not gated (repeatable)",
    )
    ap.add_argument("--dataset", default="dev", help="validation dataset (default dev)")
    ap.add_argument(
        "--independent",
        action="store_true",
        help="no source groups: every image its own group (only for unrelated images)",
    )
    a = ap.parse_args(argv)
    if bool(a.window) != bool(a.null):
        ap.error("--window and --null go together")
    if a.window and a.seed_ref:
        ap.error("--window takes its noise from --null, not --seed-ref")
    if not (gated := tuple(m for m in PERCEPTUAL if m not in a.report_only)):
        ap.error("--report-only: at least one perceptual metric must stay gated")
    pairs = [p.split("/") for p in a.null.split(",")] if a.null else []
    if any(len(p) != 2 or p[0] == p[1] for p in pairs):
        ap.error(f"--null {a.null}: X/Y pairs of two different runs, comma-separated")
    data = load_data(a.data, independent=a.independent)
    names = [
        r
        for arm in (a.ref, *a.arms, *(x for p in pairs for x in p))
        for r in arm.split("+")
    ]
    its = window_iters(a.runs, names, a.dataset, a.window) if a.window else [a.iter]

    def rows(arm: str) -> dict:
        """One run at one update as logged; else the per-image mean over the window's updates
        and the pooled runs. {} if any of them was not evaluated."""
        parts = [
            per_image(a.runs, data, run, it, a.dataset)
            for run in arm.split("+")
            for it in its
        ]
        if not all(parts):
            return {}
        return parts[0] if len(parts) == 1 else mean_rows(parts)

    at = (
        f"window {a.window} (updates {', '.join(map(str, its))})"
        if a.window
        else f"{a.iter}"
    )
    ref = rows(a.ref)
    if not ref:
        sys.exit(f"{a.ref}: no {a.dataset} rows at {at}")
    if a.window:
        rms = null_rms(rows, pairs)
        floor = taus(ref, rms)
    else:
        seed = rows(a.seed_ref) if a.seed_ref else None
        if a.seed_ref and not seed:
            sys.exit(f"{a.seed_ref}: no {a.dataset} rows at {at} for the seed spread")
        floor = floors(ref, seed)
    gated = tuple(m for m in gated if m in floor)
    print(f"## {a.dataset} at {at}: means over the {len(ref)} compared images")
    for arm in dict.fromkeys([a.ref, *([a.seed_ref] if a.seed_ref else []), *a.arms]):
        if not (r := rows(arm)):
            print(f"{arm}: not evaluated")
            continue
        means = {m: round(float(np.mean([x[m] for x in r.values()])), 4) for m in floor}
        print(f"{arm}: {means}")
    if a.window:
        print(
            "## null RMS ("
            + a.null
            + "): "
            + ", ".join(f"{m} {v:.4f}" for m, v in rms.items())
        )
        print("## tau: " + ", ".join(f"{m} {v:.4f}" for m, v in floor.items()))
    else:
        print("## floors: " + ", ".join(f"{m} {v:.4f}" for m, v in floor.items()))
    print(
        f"## gated: {', '.join(gated)}; reported: {', '.join(m for m in floor if m not in gated)}"
    )
    print(f"## paired vs {a.ref}: mean [95 % CI over source groups] status; verdict")
    for arm in a.arms:
        r = rows(arm)
        if not r:
            print(f"{arm}: not evaluated")
            continue
        p = paired(r, ref, data, tuple(floor))
        states = {m: status(m, *p[m], floor[m]) for m in floor}
        v = verdict(states, gated)
        if v == "LOSS" and "better" in (states[m] for m in gated):
            v += " (mixed)"  # not a clear loser (D10 screen)
        cells = [
            f"{m} {mean:+.4f} [{lo:+.4f}, {hi:+.4f}] {states[m]}"
            for m, (mean, lo, hi) in ((m, p[m]) for m in floor)
        ]
        print(
            f"{arm}: {v} | "
            + " | ".join(cells)
            + f" | n={p['n']}, groups={p['groups']}"
        )
    print("## diagnostics")
    for run in dict.fromkeys(r for arm in (a.ref, *a.arms) for r in arm.split("+")):
        d = diagnostics(a.runs, run)
        print(run, json.dumps(d, default=lambda x: round(float(x), 4)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
