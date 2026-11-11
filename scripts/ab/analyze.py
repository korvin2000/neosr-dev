"""A/B verdicts from scripts/ab/run_ab.py runs: dataset means, paired per-image differences against a
reference arm with a 95 % bootstrap CI over source groups, the gate (README.md) and training
diagnostics.

    python scripts/ab/analyze.py --runs RUNS --data DATA REF ARM [ARM ...]
        [--seed-ref REF2] [--iter 8000] [--dataset dev] [--independent]

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


def verdict(states: dict[str, str]) -> str:
    """LOSS if a perceptual metric is worse; WIN if >= 2 of 3 are better and none worse; else TIE."""
    perc = [states[m] for m in PERCEPTUAL if m in states]
    if "worse" in perc:
        return "LOSS"
    return "WIN" if perc.count("better") >= 2 else "TIE"


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
    if (done := runs / arm / "done.json").exists():
        out |= json.loads(done.read_text(encoding="utf-8"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="A/B verdicts with a CI over source groups"
    )
    ap.add_argument("ref", help="reference arm")
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
    ap.add_argument("--dataset", default="dev", help="validation dataset (default dev)")
    ap.add_argument(
        "--independent",
        action="store_true",
        help="no source groups: every image its own group (only for unrelated images)",
    )
    a = ap.parse_args(argv)
    data = load_data(a.data, independent=a.independent)

    def rows(arm: str) -> dict:
        return per_image(a.runs, data, arm, a.iter, a.dataset)

    ref = rows(a.ref)
    if not ref:
        sys.exit(f"{a.ref}: no {a.dataset} rows at {a.iter}")
    seed = rows(a.seed_ref) if a.seed_ref else None
    if a.seed_ref and not seed:
        sys.exit(f"{a.seed_ref}: no {a.dataset} rows at {a.iter} for the seed spread")
    floor = floors(ref, seed)
    print(f"## {a.dataset} at {a.iter}: means over the {len(ref)} compared images")
    for arm in dict.fromkeys([a.ref, *([a.seed_ref] if a.seed_ref else []), *a.arms]):
        if not (r := rows(arm)):
            print(f"{arm}: not evaluated")
            continue
        means = {m: round(float(np.mean([x[m] for x in r.values()])), 4) for m in floor}
        print(f"{arm}: {means}")
    print("## floors: " + ", ".join(f"{m} {v:.4f}" for m, v in floor.items()))
    print(f"## paired vs {a.ref}: mean [95 % CI over source groups] status; verdict")
    for arm in a.arms:
        r = rows(arm)
        if not r:
            print(f"{arm}: not evaluated")
            continue
        p = paired(r, ref, data, tuple(floor))
        states = {m: status(m, *p[m], floor[m]) for m in floor}
        cells = [
            f"{m} {v:+.4f} [{lo:+.4f}, {hi:+.4f}] {states[m]}"
            for m, (v, lo, hi) in ((m, p[m]) for m in floor)
        ]
        print(
            f"{arm}: {verdict(states)} | "
            + " | ".join(cells)
            + f" | n={p['n']}, groups={p['groups']}"
        )
    print("## diagnostics")
    for arm in dict.fromkeys([a.ref, *a.arms]):
        d = diagnostics(a.runs, arm)
        print(arm, json.dumps(d, default=lambda x: round(float(x), 4)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
