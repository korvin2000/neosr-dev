import io

import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from neosr.config import ConfigError
from neosr.models.phase import Ema, Phase, is_schedule_free, trainable
from neosr.optimizers import adamw_sf, adan_sf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class Ctx:
    def __init__(self, lq, gt) -> None:
        self.lq, self.gt, self.logs = lq, gt, {}

    def log(self, key, value) -> None:
        self.logs[key] = value.detach()


class Term:
    def __init__(self, net) -> None:
        self.net = net

    def compute(self, ctx):
        return (self.net(ctx.lq).float() - ctx.gt).abs().mean()


class Log:
    def __init__(self) -> None:
        self.msgs = []

    def warning(self, msg) -> None:
        self.msgs.append(msg)


def _net(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.Conv2d(4, 3, 1)).to(DEVICE)


def _phase(net, make_opt, *, fp16=False, scheds=(), opt_type=None, names=None):
    n, params = trainable(net)
    opt = make_opt(params)
    return Phase(
        "g",
        params,
        opt,
        [Term(net)],
        amp_dtype=torch.float16,
        use_amp=fp16,
        accumulate=1,
        grad_clip=0.0,
        schedulers=[s(opt) for s in scheds],
        names=n if names is None else names,
        opt_type=opt_type,
    )


def _train(phase, steps=3):
    g = torch.Generator(DEVICE).manual_seed(1)
    for _ in range(steps):
        lq, gt = (torch.rand(2, 3, 8, 8, generator=g, device=DEVICE) for _ in "ab")
        phase.backward(Ctx(lq, gt))
        assert phase.step()


def _roundtrip(sd):  # what the .state file does
    buf = io.BytesIO()
    torch.save(sd, buf)
    buf.seek(0)
    return torch.load(buf, weights_only=True)


def _same_opt_state(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert sa["param_groups"] == sb["param_groups"]
    assert sa["state"].keys() == sb["state"].keys()
    for k, st in sa["state"].items():
        assert st.keys() == sb["state"][k].keys()
        for name, v in st.items():
            w = sb["state"][k][name]
            assert torch.equal(v, w) if torch.is_tensor(v) else v == w, (k, name)


def _adamw(ps):
    return torch.optim.AdamW(ps, lr=1e-2)


MSLR = (lambda o: MultiStepLR(o, milestones=[2], gamma=0.5),)


def test_trainable_skips_frozen_in_order():
    net = _net()
    net[0].bias.requires_grad_(False)
    names, params = trainable(net)
    assert names == ["0.weight", "1.weight", "1.bias"]
    assert params == [net[0].weight, net[1].weight, net[1].bias]


def test_is_schedule_free():
    ps = list(_net().parameters())
    assert is_schedule_free(adamw_sf(ps))
    assert is_schedule_free(adan_sf(ps))
    assert not is_schedule_free(adan_sf(ps, schedule_free=False))
    assert not is_schedule_free(torch.optim.AdamW(ps))
    # HEAD: adamw_sf refuses schedule_free=False instead of ignoring it
    with pytest.raises(ValueError, match="always schedule-free"):
        adamw_sf(ps, schedule_free=False)


@CUDA
def test_adamw_multistep_fp16_roundtrip():
    a = _phase(_net(), _adamw, fp16=True, scheds=MSLR)
    _train(a)
    a.scaler.update(8.0)  # distinct from init_scale 32
    sd = _roundtrip(a.state_dict())
    assert sd["optimizer"]["type"] == "AdamW"
    assert sd["optimizer"]["sf"] is False
    b = _phase(_net(seed=1), _adamw, fp16=True, scheds=MSLR)
    log = Log()
    b.load_state_dict(sd, log)
    assert not log.msgs
    _same_opt_state(a.opt, b.opt)
    (sa,), (sb,) = a.schedulers, b.schedulers
    assert sb.last_epoch == sa.last_epoch == 3
    assert sb.get_last_lr() == sa.get_last_lr() == [5e-3]
    assert b.opt.param_groups[0]["lr"] == 5e-3
    assert b.scaler.get_scale() == a.scaler.get_scale() == 8.0


def test_adan_sf_roundtrip_keeps_eval_mode():
    def mk(ps):
        return adan_sf(ps, lr=1e-2)

    a = _phase(_net(), mk, opt_type="adan_sf")
    _train(a)
    a.opt.eval()
    sd = _roundtrip(a.state_dict())
    assert sd["optimizer"]["sf"] is True
    assert sd["scaler"] == {}
    b = _phase(_net(seed=1), mk, opt_type="adan_sf")
    b.load_state_dict(sd, Log())
    _same_opt_state(a.opt, b.opt)
    assert all(not g["train_mode"] for g in b.opt.param_groups)


def _raises(sd, phase, *parts):
    with pytest.raises(ConfigError) as e:
        phase.load_state_dict(sd, Log())
    msg = str(e.value)
    assert all(p in msg for p in parts), msg
    assert "pretrain_network_" in msg


def test_mismatch_raises_naming_first_difference():
    sd = _phase(_net(), _adamw, scheds=MSLR).state_dict()
    # optimizer type
    _raises(
        sd, _phase(_net(), adan_sf, scheds=MSLR), "AdamW(sf=False)", "adan_sf(sf=True)"
    )
    # same configured type, different SF mode
    sf_sd = _phase(_net(), adan_sf, opt_type="adan_sf").state_dict()
    off = _phase(
        _net(), lambda ps: adan_sf(ps, schedule_free=False), opt_type="adan_sf"
    )
    _raises(sf_sd, off, "adan_sf(sf=True)", "adan_sf(sf=False)")
    # renamed parameter
    renamed = _phase(
        _net(), _adamw, scheds=MSLR, names=["0.weight", "0.b", "1.weight", "1.bias"]
    )
    _raises(sd, renamed, "#1", "0.bias -> 0.b")
    # one fewer trainable parameter
    net = _net()
    net[1].bias.requires_grad_(False)
    _raises(sd, _phase(net, _adamw, scheds=MSLR), "#3", "1.bias -> None")
    # same count, different trainable set: positions alone would not see it
    a, b = _net(), _net()
    a[0].bias.requires_grad_(False)
    b[1].bias.requires_grad_(False)
    _raises(
        _phase(a, _adamw, scheds=MSLR).state_dict(),
        _phase(b, _adamw, scheds=MSLR),
        "#1",
        "1.weight -> 0.bias",
    )
    # scheduler type, and an added scheduler
    cos = (lambda o: CosineAnnealingLR(o, 10),)
    _raises(
        sd,
        _phase(_net(), _adamw, scheds=cos),
        "scheduler #0",
        "MultiStepLR -> CosineAnnealingLR",
    )
    _raises(
        sd,
        _phase(_net(), _adamw, scheds=MSLR + cos),
        "scheduler #1",
        "None -> CosineAnnealingLR",
    )


@CUDA
def test_scaler_mismatch_warns():
    fp32, fp16 = (_phase(_net(), _adamw, fp16=f) for f in (False, True))
    log = Log()
    fp16.load_state_dict(fp32.state_dict(), log)
    assert len(log.msgs) == 1
    assert "starts fresh" in log.msgs[0]
    fp32.load_state_dict(fp16.state_dict(), log)
    assert len(log.msgs) == 2
    assert "ignored" in log.msgs[1]


def test_state_needs_names():
    net = _net()
    _, params = trainable(net)
    kw = {
        "amp_dtype": torch.float16,
        "use_amp": False,
        "accumulate": 1,
        "grad_clip": 0.0,
    }
    p = Phase("g", params, _adamw(params), [Term(net)], **kw)
    with pytest.raises(RuntimeError, match="trainable names"):
        p.state_dict()
    sd = _phase(net, _adamw).state_dict()
    with pytest.raises(RuntimeError, match="trainable names"):
        p.load_state_dict(sd, Log())
    with pytest.raises(ValueError, match="2 names for 4"):
        Phase("g", params, _adamw(params), [Term(net)], names=["a", "b"], **kw)


def _bn_net(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4)).to(DEVICE)


@torch.no_grad()
def test_ema_state_roundtrip_then_lerps():
    net = _bn_net()
    ema = Ema(net, 0.75)
    assert ema.state_dict() is None
    ema.update()  # first update copies
    for t in (*net.parameters(), *net.buffers()):
        t.add_(1)  # live moves on; one lerp, buffers left stale on the EMA side
    ema.update()
    net[1].running_mean.add_(1)
    live, held = net.state_dict(), ema.module.state_dict()
    sd = _roundtrip(ema.state_dict())
    assert sd.keys() == held.keys()
    for k, v in sd.items():  # serialises the EMA, not the live net
        assert torch.equal(v, held[k]), k
    for k in ("0.weight", "1.weight", "1.running_mean"):
        assert not torch.equal(sd[k], live[k]), k
    net2 = _bn_net(seed=1)
    ema2 = Ema(net2, 0.75)
    ema2.load_state_dict(sd)
    assert ema2.state_dict() is not None
    for k, v in ema2.module.state_dict().items():  # exact restore, buffers included
        assert torch.equal(v, sd[k]), k
    ema2.update()  # lerp toward net2, not a copy of it
    for (k, e), live2 in zip(
        ema2.module.named_parameters(), net2.parameters(), strict=True
    ):
        assert torch.allclose(e, sd[k] + 0.25 * (live2 - sd[k])), k
        assert not torch.equal(e, live2), k
