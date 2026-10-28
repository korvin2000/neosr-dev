"""Stage 3 A1: self-describing exports, the best-G export, safetensors read, the relaxed-load
summary and the old ESRGAN layout. CPU, tiny nets."""

import logging
import sys
import types

import pytest
import torch

from neosr.archs import build_network
from neosr.config import ConfigError, validate_train
from neosr.models import checkpoint as ck
from tests.test_entry import _restore_globals as _restore_globals  # autouse fixture
from tests.test_entry import convert_py, cpu, data, test_py, toml, train_py  # noqa: F401
from tests.test_model import SF, batches, build, loader, make_opt, run

LOG = logging.getLogger("test_s3_exports")
TINY = {"type": "compact", "num_feat": 8, "num_conv": 2}


def _probe(m, values):
    """The model's psnr metric returns the next value of `values` per validation."""
    it = iter(values)
    vals: list[float] = []
    m.metrics = {"psnr": lambda *_: torch.tensor(vals[-1])}
    return lambda: vals.append(next(it))


# ------------------------------------------------------------------ export payload
def test_exports_carry_builtin_metadata_and_read_as_before(tmp_path):
    m = build(make_opt(tmp_path, d=True))
    run(m, batches(2))
    m.save()
    for name, net in (("net_g", "network_g"), ("net_d", "network_d")):
        obj = torch.load(tmp_path / "models" / f"{name}_2.pth", weights_only=True)
        assert obj["neosr"] == {net: m.opt[net], "scale": 2, "update": 2}
        assert ck.export_meta(tmp_path / "models" / f"{name}_2.pth") == obj["neosr"]
        sd = ck.read_weights(tmp_path / "models" / f"{name}_2.pth")
        assert sd.keys() == m.nets[name].state_dict().keys()


def test_export_meta_of_plain_files_is_empty(tmp_path):
    torch.save({"params": {"a": torch.ones(1)}}, tmp_path / "p.pth")
    torch.save({"a": torch.ones(1)}, tmp_path / "raw.pth")
    assert ck.export_meta(tmp_path / "p.pth") == {}
    assert ck.export_meta(tmp_path / "raw.pth") == {}
    assert ck.export_meta(tmp_path / "x.safetensors") == {}  # not opened


def test_merge_network_explicit_wins_and_a_different_type_drops_the_file(caplog):
    saved = {"type": "compact", "num_feat": 8, "num_conv": 2}
    assert ck.merge_network(saved, {}, LOG) == saved
    assert ck.merge_network(saved, {"num_conv": 3}, LOG) == saved | {"num_conv": 3}
    with caplog.at_level(logging.WARNING, logger=LOG.name):
        assert ck.merge_network(saved, {"type": "span"}, LOG) == {"type": "span"}
    assert "differs from the file's 'compact'" in caplog.text
    assert ck.merge_network(None, {"type": "span"}, LOG) == {"type": "span"}


# ------------------------------------------------------------------ val.save_best
def _best_opt(tmp_path, **kw):
    opt = make_opt(tmp_path, **kw)
    opt["val"]["save_best"] = "psnr"
    return opt


@pytest.mark.parametrize("kind", ["ema", "sf"])
def test_save_best_writes_the_evaluated_weights_only_on_improvement(tmp_path, kind):
    m = build(
        _best_opt(tmp_path, ema=0.9) if kind == "ema" else _best_opt(tmp_path, optim=SF)
    )
    step = _probe(m, [20.0, 19.0, 21.0])
    best = tmp_path / "models" / "net_g_best_valset_psnr.pth"
    run(m, batches(1))
    step()
    m.validation(loader(), m.update, None, save_img=False)
    obj = torch.load(best, weights_only=True)
    ema = m.export_state_dicts()["net_g"]  # EMA, or SF x: what validate() evaluated
    assert obj["neosr"]["update"] == 1
    assert obj["neosr"]["metric"] == {
        "dataset": "valset",
        "key": "psnr",
        "type": "calculate_psnr",
        "better": "higher",
        "value": 20.0,
    }
    assert all(torch.equal(obj["params"][k], v) for k, v in ema.items())
    run(m, batches(1, seed=1))
    step()  # worse: the file stays
    m.validation(loader(), m.update, None, save_img=False)
    assert torch.load(best, weights_only=True)["neosr"]["update"] == 1
    run(m, batches(1, seed=2))
    step()
    m.validation(loader(), m.update, None, save_img=False)
    obj = torch.load(best, weights_only=True)
    assert (obj["neosr"]["update"], obj["neosr"]["metric"]["value"]) == (3, 21.0)
    assert ck.read_weights(best).keys() == ema.keys()


def test_save_best_survives_pruning_and_resumes_its_record(tmp_path):
    opt = _best_opt(tmp_path)
    opt["logger"] = {"keep_exports": 1}
    m = build(opt)
    step = _probe(m, [20.0])
    run(m, batches(1))
    step()
    m.validation(loader(), m.update, None, save_img=False)
    m.save()
    run(m, batches(1, seed=1))
    m.save()
    models = sorted(p.name for p in (tmp_path / "models").iterdir())
    assert models == ["net_g_2.pth", "net_g_best_valset_psnr.pth"]
    # resume from update 2: a lower value than the saved record writes nothing
    opt2 = _best_opt(tmp_path)
    opt2["path"]["resume_state"] = str(tmp_path / "training_states" / "2.state")
    r = build(opt2)
    assert r.best_metric_results["valset"]["psnr"]["val"] == 20.0
    step = _probe(r, [19.0])
    run(r, batches(1, seed=2))
    step()
    r.validation(loader(), r.update, None, save_img=False)
    obj = torch.load(
        tmp_path / "models" / "net_g_best_valset_psnr.pth", weights_only=True
    )
    assert obj["neosr"]["update"] == 1


def test_a_failed_best_export_records_nothing(tmp_path, monkeypatch):
    from neosr.models import base  # noqa: PLC0415

    m = build(_best_opt(tmp_path))
    step = _probe(m, [20.0, 21.0])
    run(m, batches(1))
    step()
    m.validation(loader(), m.update, None, save_img=False)
    before = {k: dict(v) for k, v in m.best_metric_results["valset"].items()}

    def full(*_a, **_k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(base, "atomic_save", full)
    run(m, batches(1, seed=1))
    step()
    with pytest.raises(OSError, match="No space"):
        m.validation(loader(), m.update, None, save_img=False)
    assert m.best_metric_results["valset"] == before


@pytest.mark.usefixtures("cpu")
def test_the_final_state_records_the_best_of_the_final_validation(tmp_path, data):
    f = toml(tmp_path, data, total=4, save=2, val_freq=10)
    f.write_text(
        f.read_text().replace("pbar = false", 'pbar = false\nsave_best = "psnr"')
    )
    assert train_py.main(["-opt", str(f)]) == 0
    d = tmp_path / "exp" / "tiny"
    state = torch.load(d / "training_states" / "4.state", weights_only=True)
    assert state["best"]["valset"]["psnr"]["iter"] == 4
    meta = ck.export_meta(d / "models" / "net_g_best_valset_psnr.pth")
    assert meta["metric"]["value"] == state["best"]["valset"]["psnr"]["val"]


def test_validate_train_rejects_non_builtin_network_values():
    import datetime  # noqa: PLC0415

    cfg = {
        "scale": 2,
        "network_g": TINY | {"created": datetime.date(2026, 10, 28)},
        "datasets": {"train": {"patch_size": 8, "batch_size": 1}},
        "train": {"optim_g": {"type": "adamw"}},
        "logger": {"save_checkpoint_freq": 1},
    }
    with pytest.raises(ConfigError, match=r"\[network_g\] may hold only"):
        validate_train(cfg)


def test_validate_train_rejects_an_unknown_save_best_metric():
    cfg = {
        "scale": 2,
        "network_g": TINY,
        "datasets": {"train": {"patch_size": 8, "batch_size": 1}},
        "train": {"optim_g": {"type": "adamw"}},
        "logger": {"save_checkpoint_freq": 1},
        "val": {"save_best": "psnrr", "metrics": {"psnr": {"type": "calculate_psnr"}}},
    }
    with pytest.raises(ConfigError, match="did you mean 'psnr'"):
        validate_train(cfg)


# ------------------------------------------------------------------ read_weights / load_weights
def test_read_weights_safetensors_optional(tmp_path, monkeypatch):
    path = tmp_path / "m.safetensors"
    monkeypatch.setitem(sys.modules, "safetensors", None)
    monkeypatch.setitem(sys.modules, "safetensors.torch", None)
    with pytest.raises(ConfigError, match="pip install safetensors"):
        ck.read_weights(path)
    seen = []

    def load_file(p, device):
        seen.append((p, device))
        return {"module.a": torch.ones(1)}

    fake = types.ModuleType("safetensors.torch")
    fake.load_file = load_file  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "safetensors", types.ModuleType("safetensors"))
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake)
    assert list(ck.read_weights(path)) == ["a"]
    assert seen == [(str(path), "cpu")]


def test_relaxed_load_logs_one_capped_summary(caplog):
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.Conv2d(4, 3, 3))
    sd = {"0.weight": torch.zeros(4, 3, 3, 3), "1.weight": torch.zeros(5, 4, 3, 3)}
    sd |= {f"extra{i:02d}": torch.ones(1) for i in range(12)}
    with caplog.at_level(logging.INFO, logger=LOG.name):
        ck.load_weights(net, sd, strict=False, log=LOG)
    (rec,) = caplog.records
    msg = rec.getMessage()
    assert rec.levelno == logging.WARNING
    assert msg.startswith("pretrained: 1/4 network tensors loaded (25.0%)")
    assert "shape mismatch, skipped (1): 1.weight (5, 4, 3, 3) -> (3, 4, 3, 3)" in msg
    assert "missing (2): 0.bias, 1.bias" in msg
    assert "unexpected (12): extra00, " in msg
    assert "extra09, ... (+2)" in msg
    assert "extra10" not in msg
    assert "1.weight" in sd  # caller's dict untouched


def test_relaxed_full_match_logs_info(caplog):
    net = torch.nn.Conv2d(3, 3, 1)
    with caplog.at_level(logging.INFO, logger=LOG.name):
        ck.load_weights(net, net.state_dict(), strict=False, log=LOG)
    (rec,) = caplog.records
    assert rec.levelno == logging.INFO
    assert rec.getMessage() == "pretrained: 2/2 network tensors loaded (100.0%)"


# ------------------------------------------------------------------ esrgan old layout
ESRGAN = {"type": "esrgan", "num_feat": 8, "num_block": 2, "num_grow_ch": 4}


def _old_layout(sd, num_block):
    """The original ESRGAN key layout (as spandrel converts it), spelled out independently."""
    top = {"conv_first": 0, "conv_up1": 3, "conv_up2": 6, "conv_hr": 8, "conv_last": 10}
    out = {}
    for k, v in sd.items():
        mod, kind = k.rsplit(".", 1)
        if mod in top:
            out[f"model.{top[mod]}.{kind}"] = v
        elif mod == "conv_body":
            out[f"model.1.sub.{num_block}.{kind}"] = v
        else:  # body.i.rdbR.convC
            _, i, rdb, conv = mod.split(".")
            out[f"model.1.sub.{i}.RDB{rdb[3:]}.{conv}.0.{kind}"] = v
    return out


def test_esrgan_loads_the_old_community_layout(tmp_path):
    torch.manual_seed(0)
    src = build_network(ESRGAN, scale=4).eval()
    old = _old_layout(src.state_dict(), 2)
    assert "model.1.sub.1.RDB3.conv5.0.weight" in old
    torch.save(old, tmp_path / "old.pth")
    net = build_network(ESRGAN, scale=4).eval()
    ck.load_weights(net, ck.read_weights(tmp_path / "old.pth"), strict=True, log=LOG)
    assert net.state_dict().keys() == src.state_dict().keys()
    x = torch.rand(1, 3, 6, 5)
    with torch.inference_mode():
        assert torch.equal(net(x), src(x))


def test_esrgan_old_layout_relaxed_into_x2_skips_only_conv_first(caplog):
    src = build_network(ESRGAN, scale=4)
    net = build_network(ESRGAN, scale=2)
    with caplog.at_level(logging.WARNING, logger=LOG.name):
        ck.load_weights(net, _old_layout(src.state_dict(), 2), strict=False, log=LOG)
    msg = caplog.records[0].getMessage()
    assert "shape mismatch, skipped (1): conv_first.weight" in msg
    assert "missing" not in msg
    assert "unexpected" not in msg
    assert torch.equal(net.conv_last.weight, src.conv_last.weight)


def test_esrgan_old_layout_with_an_unknown_layer_fails_strict():
    old = _old_layout(build_network(ESRGAN, scale=4).state_dict(), 2)
    old["model.13.weight"] = torch.zeros(1)  # an x8 model's extra layer
    with pytest.raises(RuntimeError, match=r"Unexpected key\(s\).*model\.13\.weight"):
        build_network(ESRGAN, scale=4).load_state_dict(old)


# ------------------------------------------------------------------ convert.py / test.py
def _export(tmp_path, net=TINY, scale=2):
    torch.manual_seed(0)
    model = build_network(net, scale=scale)
    meta = {"network_g": net, "scale": scale, "update": 7}
    torch.save({"params": model.state_dict(), "neosr": meta}, tmp_path / "w.pth")
    return model


def test_convert_defaults_net_scale_and_kwargs_from_the_file(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    _export(tmp_path)
    argv = ["--input", str(tmp_path / "w.pth"), "--output", str(tmp_path / "m.onnx")]
    assert convert_py.main(argv) == 0
    assert (tmp_path / "m_fp32.onnx").is_file()


def test_convert_explicit_flags_win(tmp_path, caplog):
    from neosr.cli import convert_args  # noqa: PLC0415

    _export(tmp_path)
    args = convert_args(["--input", str(tmp_path / "w.pth")])
    convert_py.resolve_args(args)
    assert (args.network, args.scale, args.net_kw) == (
        "compact",
        2,
        {"num_feat": 8, "num_conv": 2},
    )
    net = convert_py.load_net(args)
    assert net.state_dict().keys() == build_network(TINY, scale=2).state_dict().keys()
    args = convert_args(["--input", str(tmp_path / "w.pth"), "-s", "4"])
    with caplog.at_level(logging.WARNING, logger="neosr"):
        convert_py.resolve_args(args)
    assert args.scale == 4
    assert "-s 4 overrides the file's scale 2" in caplog.text
    args = convert_args(["--input", str(tmp_path / "w.pth"), "-net", "span"])
    convert_py.resolve_args(args)
    assert (args.network, args.net_kw) == ("span", {})


def test_an_explicit_scale_replaces_the_files_native_scale_kwarg(tmp_path):
    from neosr.cli import convert_args  # noqa: PLC0415

    _export(tmp_path, net=TINY | {"upscale": 2})
    args = convert_args(["--input", str(tmp_path / "w.pth"), "-s", "4"])
    convert_py.resolve_args(args)
    assert (args.scale, args.net_kw) == (4, {"num_feat": 8, "num_conv": 2})
    assert convert_py.build_net(args)(torch.rand(1, 3, 4, 4)).shape[-1] == 16


def test_convert_refuses_a_network_named_by_the_file_before_writing(tmp_path, capsys):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    meta = {"network_g": {"type": "atd"}, "scale": 2}
    torch.save({"params": {}, "neosr": meta}, tmp_path / "w.pth")
    argv = ["--input", str(tmp_path / "w.pth"), "--output", str(tmp_path / "m.onnx")]
    assert convert_py.main(argv) == 2
    assert "cannot be exported" in capsys.readouterr().err
    assert sorted(p.name for p in tmp_path.iterdir()) == ["w.pth"]


def test_convert_without_metadata_needs_net(tmp_path, capsys):
    torch.save(
        {"params": build_network(TINY, scale=2).state_dict()}, tmp_path / "w.pth"
    )
    assert convert_py.main(["--input", str(tmp_path / "w.pth")]) == 2
    assert "pass -net" in capsys.readouterr().err


@pytest.mark.usefixtures("cpu")
def test_test_py_takes_network_options_from_the_export(tmp_path, data):  # noqa: F811
    _export(tmp_path)
    f = tmp_path / "infer.toml"
    f.write_text(f"""
name = "meta"
model_type = "image"
scale = 2
[datasets.test_1]
name = "valset"
type = "paired"
dataroot_gt = "{data / "val" / "gt"}"
dataroot_lq = "{data / "val" / "lq"}"
[val]
pbar = false
[path]
pretrain_network_g = "{tmp_path / "w.pth"}"
results_root = "{tmp_path / "results"}"
[network_g]
type = "compact"
""")
    assert test_py.main(["-opt", str(f)]) == 0
    assert len(list((tmp_path / "results" / "meta" / "valset").iterdir())) == 2
