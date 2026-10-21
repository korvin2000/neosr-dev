"""Wave E1 (design T2 §9): continuation equivalence on CPU. Run A does 6 updates. Run B does 3, saves,
and a fresh model (every stream reseeded) resumes that state, calls restore_rng() and does 3 more. Each
run's whole training state (weights and buffers, EMA, optimizers, schedulers, scalers, counters, best
metrics, every RNG stream), snapshotted right after its last update, must be bit-identical."""

import random
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from neosr.config import ConfigError
from neosr.models import build_model
from neosr.runtime import Runtime
from neosr.utils import rng as neosr_rng

CPU = Runtime(torch.device("cpu"), 0, 1, 0, False)
S, P, B = (
    2,
    16,
    2,
)  # scale, lq patch (gt 32: metagan, FDL and MS-SWD need the room), batch
SCHED = {"type": "multisteplr", "milestones": [2, 4], "gamma": 0.5}
METAGAN = {  # small: 4 stride-2 stages, spectral norm in every Down, DropPath + attention + dropout
    "type": "metagan",
    "dims": [8, 16, 32, 32],
    "blocks": [1, 1, 1, 1],
    "downs": [2, 2, 2, 2],
}


def make_opt(tmp: Path, case: str) -> dict:
    """(a) G-only L1 + FDL + MS-SWD, EMA; (b) adan_sf SF, no EMA; (c) BN G + metagan D, GAN, accumulate 2."""
    train: dict = {
        "optim_g": {"type": "adamw", "lr": 1e-3},
        "scheduler": SCHED,
        "pixel_opt": {"type": "L1Loss", "loss_weight": 1.0},
    }
    opt = {
        "name": f"e1{case}",
        "model_type": "image",
        "scale": S,
        "is_train": True,
        "network_g": {"type": "compact", "num_feat": 8, "num_conv": 2},
        "datasets": {
            "train": {"name": "t", "type": "paired", "patch_size": P, "batch_size": B}
        },
        "val": {
            "pbar": False,
            "metrics": {"psnr": {"type": "calculate_psnr", "crop_border": 0}},
        },
        "logger": {},
        "path": {
            "models": tmp / "models",
            "training_states": tmp / "training_states",
            "visualization": tmp / "vis",
        },
        "train": train,
    }
    if case == "a":
        train |= {
            "ema": 0.999,
            "fdl_opt": {"type": "fdl_loss", "model": "vgg", "num_proj": 4, "patch_size": 2, "loss_weight": 0.5},
            "msswd_opt": {"type": "msswd_loss", "num_scale": 2, "num_proj": 4, "patch_size": 3, "loss_weight": 0.5},
        }  # fmt: skip
    elif case == "b":
        train |= {
            "optim_g": {"type": "adan_sf", "lr": 1e-2, "schedule_free": True},
            "ema": 0,
        }
        del train["scheduler"]
    elif case in {"c", "c-no-d"}:
        opt["network_g"] = {
            "type": "sebica_mini"
        }  # BatchNorm: running stats are state too
        opt["datasets"]["train"]["accumulate"] = 2
        if case == "c":
            opt["network_d"] = dict(METAGAN)
            train |= {
                "optim_d": {"type": "adamw", "lr": 1e-4},
                "gan_opt": {"type": "gan_loss", "loss_weight": 0.1},
            }
    return opt


class ValSet(Dataset):
    def __init__(self) -> None:
        self.opt = {"name": "valset", "type": "paired"}
        g = torch.Generator().manual_seed(7)
        self.items = [
            (
                torch.rand(3, 12, 10, generator=g),
                torch.rand(3, 12 * S, 10 * S, generator=g),
            )
            for _ in range(2)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        lq, gt = self.items[i]
        return {"lq": lq, "gt": gt, "lq_path": f"/x/img{i}.png"}


def batches(n: int, seed: int = 0) -> list[dict]:
    g = torch.Generator().manual_seed(seed)
    return [
        {
            "lq": torch.rand(B, 3, P, P, generator=g),
            "gt": torch.rand(B, 3, P * S, P * S, generator=g),
        }
        for _ in range(n)
    ]


STREAMS = ("python", "numpy", "neosr", "torch", "loader")  # + "cuda" on a CUDA device


def build(opt: dict, rt: Runtime = CPU, seed: int = 0):
    """Seed every stream rng_state() captures; the loader generator stands in for train.py's."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 (the legacy global stream is captured state)
    neosr_rng.seed_rng(seed)
    torch.manual_seed(seed)  # + every CUDA device
    model = build_model(opt, rt)
    model.data_generator = torch.Generator().manual_seed(seed)
    return model


def step(model, data) -> None:
    """Draw from the data-side streams per micro-batch as loaders and degradations do."""
    for x in data:
        random.random(), np.random.rand(), neosr_rng.rng.random()  # noqa: NPY002
        torch.rand(1, generator=model.data_generator)
        model.feed_data(x)
        model.optimize_parameters()


def assert_same(a, b, path: str = "state") -> None:
    """Recursive bit equality: tensors by dtype + torch.equal, everything else by ==."""
    if torch.is_tensor(a):
        assert torch.is_tensor(b), path
        assert a.dtype == b.dtype, f"{path}: {a.dtype} != {b.dtype}"
        assert torch.equal(a.cpu(), b.cpu()), f"{path}: tensors differ"
    elif isinstance(a, Mapping):
        assert isinstance(b, Mapping), path
        assert a.keys() == b.keys(), f"{path}: keys {sorted(a)} != {sorted(b)}"
        for k in a:
            assert_same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list | tuple):
        assert isinstance(b, list | tuple), path
        assert len(a) == len(b), f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            assert_same(x, y, f"{path}[{i}]")
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def state(model) -> dict:
    """Everything a resume promises to continue exactly; the run record is checked by `resume()`."""
    st = model.training_state()
    del st["run"]
    return st


Hook = Callable[[object], None] | None


def continuation(
    tmp: Path,
    make: Callable[[Path], dict],
    acc: int = 1,
    mid: Hook = None,
    rt: Runtime = CPU,
    restore: bool = True,
):
    """Return (A, resumed B, A's state, B's state) after 6 updates each, each state taken right after
    that run's last update (RNG streams are process-wide). `mid(model)` runs between updates 2 and 3
    in both runs."""
    data = batches(6 * acc)
    a = build(make(tmp / "a"), rt)
    step(a, data[: 2 * acc])
    if mid is not None:
        mid(a)
    step(a, data[2 * acc :])
    sa = state(a)

    b = build(make(tmp / "b"), rt)
    step(b, data[: 2 * acc])
    if mid is not None:
        mid(b)
    step(b, data[2 * acc : 3 * acc])
    b.save()
    opt = make(tmp / "b")
    opt["path"]["resume_state"] = tmp / "b" / "training_states" / "3.state"
    r = build(
        opt, rt, seed=5
    )  # every stream perturbed: everything must come from the state
    assert (r.update, r.micro) == (3, 0)
    if restore:
        r.restore_rng()
    step(r, data[3 * acc :])
    return a, r, sa, state(r)


@pytest.fixture(autouse=True)
def deterministic():
    prev = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(prev[0], warn_only=prev[1])


def sf_x(model) -> dict[str, torch.Tensor]:
    """Oracle for the SF inference weights, from the optimizer state without touching the model:
    x = y.lerp(z, 1 - 1 / beta1) (the schedule-free eval() formula, out of place)."""
    opt = model.g.opt
    names = {p: n for n, p in model.net_g.named_parameters()}
    x = {k: v.detach().cpu().clone() for k, v in model.net_g.state_dict().items()}
    for g in opt.param_groups:
        assert g["train_mode"]
        for p in g["params"]:
            x[names[p]] = (
                p.detach().lerp(opt.state[p]["z"], 1 - 1 / g["betas"][0]).cpu()
            )
    return x


def validate_and_export(model) -> None:
    model.check_finite()
    model.validation(
        DataLoader(ValSet(), batch_size=1), model.update, None, save_img=False
    )
    model.save()  # state + exports; SF: eval_weights() swaps to x and back


def test_a_g_only_fdl_msswd_ema(tmp_path):
    a, r, sa, sr = continuation(tmp_path, lambda t: make_opt(t, "a"))
    assert a.update == r.update == 6
    assert a.ema.state_dict() is not None
    assert [s.last_epoch for s in a.g.schedulers] == [6]
    assert set(sa["rng"]) == {*STREAMS, "cuda"}
    assert_same(sa, sr)
    # the check has teeth: without restore_rng() every stream (and, via MS-SWD's projections
    # from the torch stream, the weights) ends elsewhere
    *_, sr2 = continuation(tmp_path / "x", lambda t: make_opt(t, "a"), restore=False)
    for k in STREAMS:
        with pytest.raises(AssertionError):
            assert_same(sa["rng"][k], sr2["rng"][k], k)
    with pytest.raises(AssertionError):
        assert_same(sa["nets"], sr2["nets"])


def test_b_schedule_free_with_validation_and_export(tmp_path):
    refs = []

    def mid(model):
        x, y = sf_x(model), model.net_g.state_dict()
        assert not all(
            torch.equal(x[k], v.cpu()) for k, v in y.items()
        )  # x != y: the check bites
        refs.append(x)
        validate_and_export(model)

    a, _, sa, sr = continuation(tmp_path, lambda t: make_opt(t, "b"), mid=mid)
    assert a.g.sf
    assert a.ema is None
    assert set(a.best_metric_results["valset"]) == {"psnr"}
    assert_same(sa, sr)
    for run, x in zip("ab", refs, strict=True):  # the export is x, not the live y
        exp = torch.load(tmp_path / run / "models" / "net_g_2.pth", weights_only=True)
        assert_same(exp["params"], x, f"{run}.net_g_2")


def test_c_gan_accumulate_bn_spectral_norm(tmp_path):
    a, _, sa, sr = continuation(tmp_path, lambda t: make_opt(t, "c"), acc=2)
    assert a.micro == 12
    assert any(isinstance(m, torch.nn.BatchNorm2d) for m in a.net_g.modules())
    assert any(k.endswith("weight_u") for k in a.net_d.state_dict())
    assert_same(sa, sr)


def test_d_removed_discriminator_on_resume_raises(tmp_path):
    m = build(make_opt(tmp_path, "c"))
    step(m, batches(2))
    m.save()
    opt = make_opt(tmp_path, "c-no-d")
    opt["path"]["resume_state"] = tmp_path / "training_states" / "1.state"
    with pytest.raises(ConfigError, match=r"nets \['net_d', 'net_g'\] saved"):
        build(opt)
