"""Compile parity (D2; DECISIONS D10 rule 8): one training update from a saved state, eager vs
compile = true, on one fixed batch, and a verdict.

    python scripts/compile_parity.py <config.toml> <state> <out> [--code DIR] [--modes eager,...]
        [--fallback-random] [--tol 2]

Each mode runs in a fresh process with the checkout --code (default: this repo) first on sys.path:
load_config, validate_train, setup_runtime, prepare_run, build_model (it resumes <state>, which is
only read), then one batch (drawn once by the first mode, saved as <out>/batch.pt), restore_rng(),
feed_data, optimize_parameters (with accumulate > 1, once per micro-batch, each the fixed batch).
Every mode sets `deterministic = true`. Modes:

    eager         the reference
    eager2        eager again: the run-to-run floor
    compile       compile = true
    eager_fp32    use_amp off: eager_fp32 vs eager is the bf16 noise scale
    compile_fp32  compile without bf16, compared with eager_fp32

Verdict: compile (vs eager) and compile_fp32 (vs eager_fp32) PASS when every net's update moves by
||dp_c - dp_e|| / ||dp_e|| and every loss (l_*) by its relative difference no more than --tol times
the bf16 noise scale (at least the floor): compiling may change an update as much as the precision
switch does, not more. Every mode must restore the same state bit for bit. R1/R2, D outputs and grad
norms are reported, not gated.

Random streams: inside a compiled graph inductor draws its own random numbers, so a D with Dropout or
DropPath (metagan) sees other masks when compiled. That is a different noise sample, not a numeric
bug, but its D losses and update fail the verdict. --fallback-random sets
torch._inductor.config.fallback_random in the compiled modes: they draw eager's stream, and the check
measures the numerics alone.

Everything is written below <out>: <mode>/result.pt, <mode>/child.log and the mode's run folders. A mode
whose result exists is not run again; <out>/args.json keeps one setting per <out>: the arguments, the
config's sha256, the state's size and mtime, the checkout's HEAD and the sha256 of its diff (untracked
files are not seen). A noise scale that is not finite fails. Exit 0: PASS; 1: FAIL, no verdict or a
failed mode; 2: <out> holds another setting. CUDA, one GPU job at a time.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
MODES = ("eager", "eager2", "compile", "eager_fp32", "compile_fp32")
# (mode, reference): the floor, the bf16 noise scale, then the gated pairs
PAIRS = (
    ("eager2", "eager"),
    ("eager_fp32", "eager"),
    ("compile", "eager"),
    ("compile_fp32", "eager_fp32"),
)
GATED = PAIRS[2:]
SHOWN = (
    "l_",
    "r1",
    "r2",
    "out_d",
    "grad_norm",
)  # log keys in the report; only l_* gate


def child(mode: str, out: Path) -> None:
    args = json.loads((out / "args.json").read_text(encoding="utf-8"))
    sys.path.insert(0, args["code"])
    import train  # ruff: ignore[import-outside-top-level]
    from neosr import config, models, runtime  # ruff: ignore[import-outside-top-level]

    cfg = config.load_config(
        args["config"], is_train=True, root=out / mode, debug=False
    )
    cfg["deterministic"] = True
    cfg["compile"] = mode.startswith("compile")
    if mode.endswith("fp32"):
        cfg["use_amp"] = cfg["bfloat16"] = False
    if cfg["compile"] and args["fallback_random"]:
        from torch._inductor import config as inductor  # ruff: ignore[import-outside-top-level, import-private-name]

        inductor.fallback_random = True
    config.validate_train(cfg)
    cfg["path"]["resume_state"] = args["state"]  # only read (torch.load)
    rt = runtime.setup_runtime(cfg)
    train.prepare_run(cfg, ["compile_parity.py", mode])
    loader = train.build_loaders(cfg, rt)[0]
    model = models.build_model(cfg, rt)
    model.data_generator = loader.generator
    batch_file = out / "batch.pt"
    if batch_file.exists():
        batch = torch.load(batch_file, weights_only=True)
    else:
        batch = {k: v for k, v in next(iter(loader)).items() if torch.is_tensor(v)}
        torch.save(batch, batch_file)
    model.restore_rng()  # the last call before the loop, as in train.train

    def snap() -> dict:
        ema = None if model.ema is None else model.ema.module
        nets = {"G": model.net_g, "D": model.net_d, "EMA": ema}
        return {
            k: {n: p.detach().float().cpu().clone() for n, p in m.named_parameters()}
            for k, m in nets.items()
            if m is not None
        }

    before = snap()
    for _ in range(
        model.accumulate
    ):  # every micro-batch of the window is the fixed batch
        model.feed_data(batch)
        closed = model.optimize_parameters()
    assert closed, "the update window did not close"
    if rt.device.type == "cuda":
        torch.cuda.synchronize()
    after = snap()
    logs = {k: float(v) for k, v in model.get_current_log().items()}
    print(json.dumps({"mode": mode, "update": model.update, **logs}))
    torch.save(
        {"logs": logs, "before": before, "after": after}, out / mode / "result.pt"
    )


def log_diffs(ref: dict, res: dict) -> dict[str, tuple[float, float, float]]:
    """Per log key of the reference: (reference, other, relative difference); missing is nan."""
    out = {}
    for k, a in sorted(ref["logs"].items()):
        b = res["logs"].get(k, float("nan"))
        out[k] = (a, b, abs(a - b) / max(abs(a), 1e-12))
    return out


def param_diffs(ref: dict, res: dict) -> dict[str, dict]:
    """Per net, the other update dp_m against the reference's dp_e: same_start (restored states
    bitwise equal), upd = max|dp_e|, maxabs = max|dp_m - dp_e|, rho = ||dp_m - dp_e|| / ||dp_e||,
    elem = max |dp_m - dp_e| / |dp_e| over |dp_e| > 1e-3 upd."""
    out = {}
    for net, b0 in ref["before"].items():
        b1 = res["before"][net]
        de = torch.cat([(ref["after"][net][n] - b0[n]).flatten() for n in b0])
        dm = torch.cat([(res["after"][net][n] - b1[n]).flatten() for n in b0])
        diff = (dm - de).abs()
        upd = de.abs().max().item()
        big = de.abs() > 1e-3 * upd
        out[net] = {
            "same_start": all(torch.equal(b0[n], b1[n]) for n in b0),
            "upd": upd,
            "maxabs": diff.max().item(),
            "rho": (dm - de).norm().item() / max(de.norm().item(), 1e-30),
            "elem": (diff[big] / de.abs()[big]).max().item() if big.any() else 0.0,
        }
    return out


def loss_scale(ref: dict, res: dict) -> float:
    return max(
        (r for k, (_, _, r) in log_diffs(ref, res).items() if k.startswith("l_")),
        default=0.0,
    )


def verdict(res: dict[str, dict], tol: float) -> tuple[bool | None, list[str]]:
    """PASS (True) or FAIL (False) with the reasons; None without eager, eager_fp32 and compile.
    Bound per net: tol x max(rho of eager_fp32, rho of eager2); per loss: tol x the largest l_*
    relative difference of those two pairs."""
    if not {"eager", "eager_fp32", "compile"} <= res.keys():
        return None, ["no verdict: needs eager, eager_fp32 and compile"]
    refs = [m for m in ("eager_fp32", "eager2") if m in res]
    rhos = {
        net: [param_diffs(res["eager"], res[m])[net]["rho"] for m in refs]
        for net in res["eager"]["before"]
    }
    lrels = [loss_scale(res["eager"], res[m]) for m in refs]
    if not all(map(math.isfinite, [*lrels, *(r for v in rhos.values() for r in v)])):
        return False, [f"noise scale not finite: rho {rhos}, losses {lrels}"]
    scale, lscale = {net: max(v) for net, v in rhos.items()}, max(lrels)
    fails = [
        f"{m}: {net} restored state differs from {r}"
        for m, r in PAIRS
        if m in res
        for net, d in param_diffs(res[r], res[m]).items()
        if not d["same_start"]
    ]
    for m, r in GATED:
        if m not in res:
            continue
        for net, d in param_diffs(res[r], res[m]).items():
            if not d["rho"] <= tol * scale[net]:  # nan fails
                fails.append(
                    f"{m}: {net} rho {d['rho']:.2e} > {tol:g} x {scale[net]:.2e}"
                )
        for k, (_, _, rel) in log_diffs(res[r], res[m]).items():
            if k.startswith("l_") and not rel <= tol * lscale:
                fails.append(f"{m}: {k} rel {rel:.2e} > {tol:g} x {lscale:.2e}")
    if fails:
        return False, fails
    rhos = ", ".join(f"{net} {s:.2e}" for net, s in scale.items())
    return True, [f"bounds: {tol:g} x (rho {rhos}; losses {lscale:.2e})"]


def report(res: dict[str, dict], tol: float) -> tuple[bool | None, list[str]]:
    lines = []
    for m, r in PAIRS:
        if m not in res or r not in res:
            continue
        lines.append(f"-- {m} vs {r}: logs (key: {r}, {m}, rel diff)")
        lines += [
            f"   {k:22s} {a: .6e} {b: .6e} {rel:.2e}"
            for k, (a, b, rel) in log_diffs(res[r], res[m]).items()
            if k.startswith(SHOWN)
        ]
        lines.append(
            f"-- {m} vs {r}: params (net: max|dp|, max|dp_m - dp_e|, ratio, rho, max elem)"
        )
        lines += [
            f"   {net:4s} {d['upd']:.3e} {d['maxabs']:.3e} "
            f"{d['maxabs'] / max(d['upd'], 1e-30):.2e} {d['rho']:.2e} {d['elem']:.2e}"
            for net, d in param_diffs(res[r], res[m]).items()
        ]
    ok, why = verdict(res, tol)
    head = {True: "PASS", False: "FAIL", None: "NO VERDICT"}[ok]
    return ok, [*lines, f"== {head}", *(f"   {w}" for w in why)]


def code_id(code: Path) -> list[str]:
    """The checkout's HEAD and the sha256 of its diff of tracked files (empty HEAD outside git)."""

    def git(*cmd: str) -> str:
        argv = ["git", "-C", str(code), *cmd]
        return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            argv, capture_output=True, text=True, check=False
        ).stdout

    return [
        git("rev-parse", "HEAD").strip(),
        hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
    ]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--child"]:
        child(argv[1], Path(argv[2]))
        return 0
    ap = argparse.ArgumentParser(description="one update eager vs compiled: a verdict")
    ap.add_argument("config", type=Path, help="the training TOML")
    ap.add_argument("state", type=Path, help="the <update>.state to start from (read)")
    ap.add_argument("out", type=Path, help="folder for batch, results and run folders")
    ap.add_argument("--code", type=Path, default=REPO, help="checkout (default: this)")
    ap.add_argument(
        "--modes", default=",".join(MODES), help="comma list of " + ",".join(MODES)
    )
    ap.add_argument(
        "--fallback-random",
        action="store_true",
        help="compiled modes draw eager's Dropout/DropPath stream (inductor fallback_random)",
    )
    ap.add_argument("--tol", type=float, default=2.0, help="x the bf16 noise scale")
    a = ap.parse_args(argv)
    modes = a.modes.split(",")
    if bad := set(modes) - set(MODES):
        ap.error(f"unknown modes {sorted(bad)}")
    st = a.state.stat()
    args = {
        "config": str(a.config.resolve()),
        "state": str(a.state.resolve()),
        "code": str(a.code.resolve()),
        "fallback_random": a.fallback_random,
        "inputs": {  # what the results were measured on: an edit in place is another setting
            "config_sha256": hashlib.sha256(a.config.read_bytes()).hexdigest(),
            "state_size_mtime": [st.st_size, st.st_mtime_ns],
            "code": code_id(a.code),
        },
    }
    a.out.mkdir(parents=True, exist_ok=True)
    out, args_file = a.out.resolve(), a.out / "args.json"
    if args_file.exists() and json.loads(args_file.read_text("utf-8")) != args:
        print(
            f"{args_file}: results of another setting; use another out", file=sys.stderr
        )
        return 2
    args_file.write_text(json.dumps(args, indent=1), encoding="utf-8")
    env = os.environ | {"PYTHONHASHSEED": "0"}
    for mode in modes:
        run = out / mode
        if (run / "result.pt").exists():
            continue
        run.mkdir(exist_ok=True)
        r = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [sys.executable, __file__, "--child", mode, str(out)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        (run / "child.log").write_text(r.stdout + r.stderr, encoding="utf-8")
        if r.returncode:
            print(f"{mode}: FAILED (exit {r.returncode}), {run / 'child.log'}")
            print(r.stderr[-4000:])
            return 1
    res = {
        m: torch.load(f, weights_only=True)
        for m in MODES
        if (f := out / m / "result.pt").exists()
    }
    ok, lines = report(res, a.tol)
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
