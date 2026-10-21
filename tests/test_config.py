import tomllib
import warnings
from pathlib import Path

import pytest

from neosr.config import ConfigError, ConfigWarning, load_config, validate_train

ROOT = Path(__file__).resolve().parents[1]
OPTIONS = sorted((ROOT / "options").rglob("*.toml"))

BASE = """
[network_g]
type = "esrgan"

[datasets.train]
type = "paired"
dataroot_gt = "~/gt"
dataroot_lq = "/data/lq"
patch_size = 32
batch_size = 2

[datasets.val]
name = "val"
type = "paired"
dataroot_gt = "/data/val/gt"
dataroot_lq = "/data/val/lq"

[train.optim_g]
type = "adan_sf"
lr = 1e-3

[logger]
total_iter = 10
"""
HEAD = 'name = "t"\nmodel_type = "image"\nscale = 4\n'


def write(
    tmp_path: Path, body: str = BASE, head: str = HEAD, name: str = "c.toml"
) -> Path:
    path = tmp_path / name
    path.write_text(head + body, encoding="utf-8")
    return path


def load(path: Path, *, is_train: bool = True, root: Path | None = None, **kw) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConfigWarning)
        return load_config(path, is_train=is_train, root=root or path.parent, **kw)


@pytest.mark.parametrize(
    "path", OPTIONS, ids=lambda p: str(p.relative_to(ROOT / "options"))
)
def test_shipped_configs_load_without_warnings(path: Path) -> None:
    is_train = "train" in tomllib.loads(path.read_text("utf-8"))
    cfg = load(path, is_train=is_train, root=ROOT)
    if is_train:
        validate_train(cfg)


def test_base_config_is_clean_and_pure(tmp_path: Path) -> None:
    cfg = load(write(tmp_path))
    validate_train(cfg)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "c.toml"
    ]  # no directories created


def test_unknown_key_warns_with_suggestion(tmp_path: Path) -> None:
    path = write(
        tmp_path, BASE.replace("patch_size = 32", "patch_size = 32\npatch_sizee = 3")
    )
    with pytest.warns(
        ConfigWarning, match=r"datasets\.train\.patch_sizee.*did you mean 'patch_size'"
    ):
        load_config(path, is_train=True, root=tmp_path)


def test_unknown_root_key_warns(tmp_path: Path) -> None:
    with pytest.warns(ConfigWarning, match="did you mean 'use_amp'"):
        load_config(
            write(tmp_path, head=HEAD + "use_amps = true\n"),
            is_train=True,
            root=tmp_path,
        )


@pytest.mark.parametrize(
    ("head", "body", "match"),
    [
        (
            HEAD.replace("scale = 4", "scale = true"),
            BASE,
            "scale must be int, got bool",
        ),
        (HEAD.replace("scale = 4\n", ""), BASE, "missing required key 'scale'"),
        (
            HEAD,
            BASE.replace("batch_size = 2", 'batch_size = "2"'),
            "batch_size must be int",
        ),
        (
            HEAD,
            BASE.replace("[logger]", "[logger]\nprint_freq = 1.5"),
            "print_freq must be int",
        ),
        (HEAD + "manual_seed = false\n", BASE, "manual_seed must be int, got bool"),
        (HEAD, BASE + "\n[train]\nsam = 'fsam'\n", "SAM/FSAM was removed"),
        (
            HEAD,
            BASE.replace("lr = 1e-3", "lr = 1e-3\n[train.gw_opt]\ntype = 'gw_loss'"),
            "gw_loss was removed",
        ),
        (
            HEAD,
            BASE.replace(
                "[logger]", "[path]\nignore_resume_networks = ['network_g']\n[logger]"
            ),
            "removed",
        ),
        (HEAD, BASE.replace("[datasets.val]", "[datasets.valid]"), "unknown phase"),
        (
            HEAD,
            BASE + '\n[datasets.train_2]\ntype = "paired"\n',
            r"\[datasets\.train_2\] unknown phase",
        ),
        (HEAD + 'network_d = "metagan"\n', BASE, "network_d must be a table"),
        (HEAD, "name = = 1\n", None),
    ],
)
def test_config_errors(tmp_path: Path, head: str, body: str, match: str | None) -> None:
    with pytest.raises(ConfigError, match=match):
        load(write(tmp_path, body, head))


def test_float_accepts_int_but_not_bool(tmp_path: Path) -> None:
    body = BASE.replace("[train.optim_g]", "[train]\nema = 1\n\n[train.optim_g]")
    assert load(write(tmp_path, body))["train"]["ema"] == 1
    with pytest.raises(ConfigError, match="ema must be float"):
        load(write(tmp_path, body.replace("ema = 1", "ema = true")))


def test_malformed_toml_and_wrong_suffix(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"c\.toml"):
        load(write(tmp_path, "[network_g\n"))
    with pytest.raises(ConfigError, match=r"\.toml"):
        load(write(tmp_path, name="c.yml"))
    with pytest.raises(ConfigError, match=r"missing\.toml"):
        load(tmp_path / "missing.toml")


def test_degradations_merge_into_train_and_win(tmp_path: Path) -> None:
    body = BASE.replace(
        "batch_size = 2", "batch_size = 2\nsinc_prob = 0.5\nnoise_range = [0, 2]"
    )
    body += "\n[degradations]\nsinc_prob = 0.1\nnoise_range = [0, 2]\njpeg_range = [40, 95]\n"
    with pytest.warns(ConfigWarning, match="'sinc_prob' is set in") as record:
        cfg = load_config(write(tmp_path, body), is_train=True, root=tmp_path)
    assert len(record) == 1  # equal values (noise_range) do not warn
    train = cfg["datasets"]["train"]
    assert (train["sinc_prob"], train["jpeg_range"], train["noise_range"]) == (
        0.1,
        [40, 95],
        [0, 2],
    )
    assert "jpeg_range" not in cfg["datasets"]["val"]


def test_unknown_degradation_key_warns(tmp_path: Path) -> None:
    with pytest.warns(ConfigWarning, match="did you mean 'sinc_prob'"):
        load_config(
            write(tmp_path, BASE + "\n[degradations]\nsinc_pro = 0.1\n"),
            is_train=True,
            root=tmp_path,
        )


@pytest.mark.parametrize("key", ["blur_kernel_size", "blur_kernel_size2"])
def test_removed_blur_kernel_size_warns(tmp_path: Path, key: str) -> None:
    with pytest.warns(ConfigWarning, match=rf"degradations\.{key}\b"):
        cfg = load_config(
            write(tmp_path, BASE + f"\n[degradations]\n{key} = 7\n"),
            is_train=True,
            root=tmp_path,
        )
    assert key not in cfg["datasets"]["train"]


def test_unknown_degradation_key_does_not_overwrite_train(tmp_path: Path) -> None:
    with pytest.warns(ConfigWarning, match="degradations.(scale|patch_size)") as record:
        cfg = load_config(
            write(tmp_path, BASE + "\n[degradations]\nscale = 8\npatch_size = 1\n"),
            is_train=True,
            root=tmp_path,
        )
    train = cfg["datasets"]["train"]
    assert (train["scale"], train["patch_size"]) == (4, 32)
    assert len(record) == 2


def test_unknown_path_key_is_not_expanded(tmp_path: Path) -> None:
    body = BASE + '\n[path]\nresume_states = false\npretrain_network_g = "~/g.pth"\n'
    with pytest.warns(ConfigWarning, match="did you mean 'resume_state'"):
        cfg = load_config(write(tmp_path, body), is_train=True, root=tmp_path)
    assert cfg["path"]["resume_states"] is False
    assert cfg["path"]["pretrain_network_g"] == str(Path("~/g.pth").expanduser())


def test_absent_optional_sections_stay_absent(tmp_path: Path) -> None:
    cfg = load(write(tmp_path))
    for key in (
        "network_d",
        "val",
        "manual_seed",
        "use_amp",
        "num_gpu",
        "dist_params",
        "degradations",
    ):
        assert key not in cfg
    assert "optim_d" not in cfg["train"]
    assert "pretrain_network_d" not in cfg["path"]


def test_derived_keys(tmp_path: Path) -> None:
    cfg = load(write(tmp_path), auto_resume=True)
    assert (cfg["is_train"], cfg["auto_resume"], cfg["root_path"]) == (
        True,
        True,
        str(tmp_path),
    )
    assert cfg["opt_path"] == str(tmp_path / "c.toml")
    train, val = cfg["datasets"]["train"], cfg["datasets"]["val"]
    assert (train["phase"], val["phase"], train["scale"], val["scale"]) == (
        "train",
        "val",
        4,
        4,
    )
    assert train["dataroot_gt"] == str(Path("~/gt").expanduser())
    exp = tmp_path / "experiments" / "t"
    assert cfg["path"] == {
        "experiments_root": exp,
        "models": exp / "models",
        "log": exp,
        "training_states": exp / "training_states",
        "visualization": exp / "visualization",
    }


def test_experiments_root_is_honoured(tmp_path: Path) -> None:
    body = BASE.replace(
        "[logger]", f"[path]\nexperiments_root = '{tmp_path / 'runs'}'\n[logger]"
    )
    cfg = load(write(tmp_path, body))
    assert cfg["path"]["experiments_root"] == tmp_path / "runs" / "t"
    assert cfg["path"]["models"] == tmp_path / "runs" / "t" / "models"


def test_inference_paths_and_test_phase(tmp_path: Path) -> None:
    body = '[network_g]\ntype = "esrgan"\n[datasets.test_1]\nname = "a"\ntype = "single"\ndataroot_lq = "/x"\n'
    cfg = load(write(tmp_path, body), is_train=False)
    res = tmp_path / "experiments" / "results" / "t"
    assert cfg["path"] == {"results_root": res, "log": res, "visualization": res}
    assert cfg["datasets"]["test_1"]["phase"] == "test"


def test_debug_mode(tmp_path: Path) -> None:
    cfg = load(write(tmp_path, BASE + "\n[val]\nval_freq = 1000\n"), debug=True)
    assert cfg["name"] == "debug_t"
    assert cfg["path"]["experiments_root"].name == "debug_t"
    assert (
        cfg["val"]["val_freq"],
        cfg["logger"]["print_freq"],
        cfg["logger"]["save_checkpoint_freq"],
    ) == (8, 1, 8)


@pytest.mark.parametrize(
    ("head", "body", "match"),
    [
        (HEAD.replace("scale = 4", "scale = 5"), BASE, "scale=5"),
        (
            HEAD,
            BASE.replace("[network_g]", "[network_d]\ntype = 'metagan'\n[network_g]"),
            "train.optim_d is not",
        ),
        (HEAD, BASE + "\n[train.optim_d]\ntype = 'adan_sf'\n", "network_d is not"),
        (
            HEAD,
            BASE.replace("[datasets.train]", "[datasets.val_2]"),
            r"\[datasets.train\]",
        ),
        (HEAD, BASE.replace("[datasets.val]", "[datasets.test_1]"), "test datasets"),
        (HEAD, BASE.replace("[train.optim_g]", "[train.pixel_opt]"), "optim_g"),
        (HEAD + "bfloat16 = true\n", BASE, "use_amp"),
        (
            HEAD,
            BASE.replace("patch_size = 32", "patch_size = 0"),
            "patch_size must be positive",
        ),
        (HEAD, BASE.replace("patch_size = 32\n", ""), "patch_size is required"),
        (
            HEAD,
            BASE.replace("batch_size = 2", "batch_size = 2\naccumulate = 0"),
            "accumulate",
        ),
        (
            HEAD,
            BASE.replace(
                "batch_size = 2",
                "batch_size = 2\nnum_worker_per_gpu = 0\npersistent_workers = true",
            ),
            "persistent_workers",
        ),
        (
            HEAD,
            BASE.replace(
                "[train.optim_g]", "[train]\nmax_skip_streak = 0\n[train.optim_g]"
            ),
            "max_skip_streak",
        ),
        (HEAD, BASE + "\n[val]\ntile = 32\n", "need tile > 2"),  # default overlap 16
        (HEAD, BASE + "\n[val]\ntile = 16\ntile_overlap = 8\n", "need tile > 2"),
        (HEAD, BASE + "\n[val]\ntile = 16\ntile_overlap = -1\n", "need tile > 2"),
    ],
)
def test_validate_train_errors(
    tmp_path: Path, head: str, body: str, match: str
) -> None:
    cfg = load_config(write(tmp_path, body, head), is_train=True, root=tmp_path)
    with pytest.raises(ConfigError, match=match):
        validate_train(cfg)


@pytest.mark.parametrize(
    "val",
    [
        "",
        "tile = 0\n",
        "tile = 33\n",
        "tile = 17\ntile_overlap = 8\n",
        "tile = 8\ntile_overlap = 0\n",
    ],
)
def test_validate_train_accepts_valid_tiling(tmp_path: Path, val: str) -> None:
    validate_train(load(write(tmp_path, BASE + "\n[val]\n" + val)))
