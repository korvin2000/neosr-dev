"""Single-image super-resolution trainer (design T2 §3, §6, §8.4; T1 §4.5)."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from tqdm import tqdm

from neosr.archs import build_network
from neosr.config import ConfigError
from neosr.data.augmentations import apply_augment
from neosr.losses import build_losses
from neosr.losses.base import Context
from neosr.models.base import base, build_optimizer, build_schedulers
from neosr.models.checkpoint import cpu_copy, load_weights, read_weights
from neosr.models.phase import Ema, LossMeter, Phase, no_sync, trainable
from neosr.models.tiling import infer_tiled
from neosr.runtime import Runtime
from neosr.utils import get_root_logger, imwrite, tc, tensor2img


class image(base):
    """Paired SR training and inference. `otf` overrides only how a training batch is made."""

    def __init__(self, opt: dict[str, Any], runtime: Runtime) -> None:
        super().__init__(opt, runtime)
        # model-side memory format follows AMP, as before; training inputs now match it (T4 P6)
        self.mf = torch.channels_last if opt.get("use_amp") else torch.contiguous_format
        resuming = self.is_train and opt["path"].get("resume_state") is not None
        self.net_g = self._build("network_g", resuming)
        self._prepare_qat()  # topology is final after this line
        self.nets = {"net_g": self.net_g}
        if not self.is_train:  # test.py: eager, no wrapper, D, EMA, optimizer or loss
            return

        tr, ds = opt["train"], opt["datasets"]["train"]
        if opt.get("network_d") is not None:
            # D is always in train mode (T1 D10)
            self.net_d = self._build("network_d", resuming).train()
            self.nets["net_d"] = self.net_d
        self.net_g.train()
        ema = tr.get("ema", 0)  # deep copy, initialised at the first update
        self.ema = Ema(self.net_g, ema) if ema > 0 else None
        g_terms, d_terms = build_losses(
            tr, scale=self.scale, device=self.device, has_d=self.net_d is not None
        )
        self.accumulate = ds.get("accumulate", 1)
        amp = {
            "amp_dtype": torch.bfloat16 if opt.get("bfloat16") else torch.float16,
            "use_amp": bool(opt.get("use_amp")),
            "accumulate": self.accumulate,
            "grad_clip": 1.0 if tr.get("grad_clip", True) else 0.0,
            "max_skip_streak": tr.get("max_skip_streak", 10),
        }
        self.phases = {"net_g": self._phase("g", self.net_g, g_terms, amp)}
        if self.net_d is not None:
            self.phases["net_d"] = self._phase("d", self.net_d, d_terms, amp)
        self.g, self.d = self.phases["net_g"], self.phases.get("net_d")
        self.meter = LossMeter()
        self.clamp = tr.get("clamp", True)
        self.aug, self.aug_prob = ds.get("augmentation"), ds.get("aug_prob")
        if self.aug is not None and list(self.aug) == ["none"]:
            self.aug = None

        if resuming:
            self.resume(Path(opt["path"]["resume_state"]))
        # last: DDP broadcasts the loaded weights; compiled graphs see them in place
        self.net_g_fwd = self.wrap(self.net_g)
        # D: 2 forwards before 1 backward; broadcasting BN buffers in between would fail
        self.net_d_fwd = self.wrap(self.net_d, broadcast_buffers=False)
        # G's adversarial pass runs D without a DDP reducer
        if self.net_d is None or not self.rt.distributed:
            self.net_d_adv = self.net_d_fwd
        elif opt.get("compile"):
            mode = opt.get("compile_mode", "default")
            self.net_d_adv = torch.compile(self.net_d, mode=mode, dynamic=False)
        else:
            self.net_d_adv = self.net_d

    # ---------------------------------------------------------------- construction
    def _build(self, key: str, resuming: bool) -> nn.Module:
        p, s = self.opt["path"], key[-1]  # "g" | "d"
        scale = self.scale if key == "network_g" else None
        net = build_network(self.opt[key], scale=scale, section=key)
        net = net.to(self.device, memory_format=self.mf)
        if (path := p.get(f"pretrain_network_{s}")) is not None and not resuming:
            log = get_root_logger()
            log.info(f"loading {key} from {path}")
            sd = read_weights(path, p.get(f"param_key_{s}"))
            load_weights(net, sd, p.get(f"strict_load_{s}", True), log)
        self.log_network(net, key)
        return net

    def _prepare_qat(self) -> None:
        """T5 slot (T2 §3 step 3, filled in step C5): prepare QAT on the bare `self.net_g` after the
        float load and load a QAT-kind pretrain. Everything that captures parameter identity (EMA,
        optimizers, wrappers) is built after this call."""
        if (self.opt.get("train") or {}).get("enable_qat"):
            msg = "train.enable_qat is not supported yet; set it to false to train in float"
            raise ConfigError(msg)

    def _phase(
        self, name: str, net: nn.Module, terms: list, amp: dict[str, Any]
    ) -> Phase:
        section = f"train.optim_{name}"
        names, params = trainable(net)
        kind, opt = build_optimizer(self.opt["train"][f"optim_{name}"], params, section)
        scheds = build_schedulers(self.opt["train"].get("scheduler"), opt)
        return Phase(
            name,
            params,
            opt,
            terms,
            **amp,
            schedulers=scheds,
            names=names,
            opt_type=kind,
        )

    # ---------------------------------------------------------------- training
    @torch.no_grad()
    def feed_data(self, data: dict[str, Any]) -> None:
        """Paired training batch; `otf` synthesises lq instead."""
        lq = data["lq"].to(self.device, non_blocking=True)
        gt = data["gt"].to(self.device, non_blocking=True)
        self._set_batch(lq, gt)

    def _set_batch(self, lq: Tensor, gt: Tensor) -> None:
        if self.aug is not None:
            gt, lq = apply_augment(
                gt, lq, scale=self.scale, augs=self.aug, prob=self.aug_prob
            )
        self.lq = lq.contiguous(memory_format=self.mf)
        self.gt = gt.contiguous(memory_format=self.mf)

    def optimize_parameters(self) -> bool:
        """One micro-batch. Returns True when it closed an update window (then `self.update` grew)."""
        boundary = (self.micro + 1) % self.accumulate == 0
        with no_sync(self.net_g_fwd, boundary):  # forward + losses + backward
            with torch.autocast(**self.g.autocast):
                sr = self.net_g_fwd(self.lq)
                if self.clamp:
                    sr = sr.clamp(1 / 255, 1)
            ctx = Context(
                sr=sr,
                gt=self.gt,
                lq=self.lq,
                it=self.update + 1,  # the update being computed
                net_d=self.net_d_fwd,
                net_d_adv=self.net_d_adv,
            )
            self.g.backward(ctx)
        if self.d is not None:
            with no_sync(self.net_d_fwd, boundary):
                self.d.backward(ctx)
        self.meter.add(ctx.logs)
        self.micro += 1
        if not boundary:
            return False
        ok = {self.g: self.g.step()}
        if ok[self.g] and self.ema is not None:
            self.ema.update()
        if self.d is not None:
            ok[self.d] = self.d.step()
        # the mean over successful updates: a skipped one adds nothing
        self.meter.add({f"grad_norm_{p.name}": p.grad_norm for p, s in ok.items() if s})
        self.update += 1
        return True

    def get_current_log(self) -> dict[str, float]:
        """Window means since the last call (collective under DDP); raises on a non-finite value."""
        out = self.meter.read()
        for p in self.phases.values():
            out[f"skipped_{p.name}"], p.skipped = p.skipped, 0
        return out

    def check_finite(self) -> None:
        """All ranks, before validation and save: raises before bad state reaches disk or metrics."""
        self.meter.read(reset=False)
        nets = [
            self.net_g,
            *([self.ema.module] if self.ema else []),
            *([self.net_d] if self.net_d else []),
        ]
        ts = [
            t
            for n in nets
            for t in (*n.parameters(), *n.buffers())
            if t.is_floating_point()
        ]
        bad = (~torch.stack(torch._foreach_norm(ts)).isfinite().all()).int()
        # buffers are rank-local: one bad rank stops every rank
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(bad, op=torch.distributed.ReduceOp.MAX)
        if bool(bad):
            msg = "non-finite weights or buffers; the last committed checkpoint stands"
            raise FloatingPointError(msg)

    def current_lr(self) -> list[float]:
        return [g["lr"] for g in self.g.opt.param_groups]

    # ---------------------------------------------------------------- evaluation
    @contextmanager
    def eval_weights(self) -> Generator[nn.Module]:
        """The generator to evaluate, in eval mode. Restores what it changed: SF params, optimizer
        train_mode flags, per-module training flags. Arch-internal eval caches (SPAN eval_conv) are
        recomputed by the arch on every eval forward. Never entered inside inference_mode; never saves."""
        if self.ema is not None:
            yield self.ema.module  # permanently eval, requires_grad False
            return
        # phases == {} on the inference path
        net, phase = self.net_g, self.phases.get("net_g")
        was, modes = net.training, [(m, m.training) for m in net.modules()]
        snap = flags = None
        try:
            if phase is not None and phase.sf:
                flags = [g["train_mode"] for g in phase.opt.param_groups]
                snap = [p.detach().clone() for p in net.parameters()]
                phase.opt.eval()  # p <- x
            net.eval()
            self._evaluating = True
            yield net
        finally:
            self._evaluating = False
            net.train(was)  # runs arch train() overrides
            for m, t in modes:  # then the exact per-module flags
                m.training = t
            # never opt.train(): a second lerp is not a bit-inverse
            if snap is not None:
                with torch.no_grad():
                    torch._foreach_copy_(list(net.parameters()), snap)
                for g, f in zip(phase.opt.param_groups, flags, strict=True):  # type: ignore[union-attr,arg-type]
                    g["train_mode"] = f

    def validate(self, loader, current_iter, tb_logger, save_img: bool) -> None:
        """Output contract of the old nondist_validation (T2 §6.3); rank 0, called by validation()."""
        ds, val = loader.dataset.opt, self.opt.get("val") or {}
        name, vis = ds["name"], Path(self.opt["path"]["visualization"])
        metrics = {} if ds["type"] == "single" else self.metrics
        sums, n = dict.fromkeys(metrics, 0.0), 0
        pbar = (
            tqdm(total=len(loader), unit="image", colour="green", ascii=" >=")
            if val.get("pbar", True)
            else None
        )
        save_tb = self.is_train and (self.opt.get("logger") or {}).get(
            "save_tb_img", False
        )
        with self.eval_weights() as net:  # one swap per validation
            for data in loader:
                img = Path(data["lq_path"][0]).stem
                if pbar is not None:  # the image being inferred, not the last one done
                    pbar.set_description(f"{tc.light_green}Inferring on {img}{tc.end}")
                with torch.inference_mode():
                    lq = data["lq"].to(self.device, non_blocking=True)
                    sr = (
                        net(lq)
                        if self.tile <= 0
                        else infer_tiled(
                            net, lq, self.scale, self.tile, self.tile_overlap
                        )
                    )
                sr_img = tensor2img([sr])  # one host copy
                if save_img:
                    imwrite(sr_img, str(self._img_path(vis, name, img, current_iter)))
                    lq_path = vis / img / f"{img}_lq.png"
                    if val.get("save_lq", True) and not lq_path.exists():
                        imwrite(tensor2img([data["lq"]]), str(lq_path))
                if save_tb and tb_logger is not None:
                    tb_logger.add_image(
                        f"{img}/{current_iter}",
                        tensor2img([sr], rgb2bgr=False),
                        global_step=current_iter,
                        dataformats="HWC",
                    )
                if metrics:
                    gt_img = tensor2img([data["gt"]])
                    for (
                        k,
                        f,
                    ) in metrics.items():  # float(): numpy scalars fail weights_only
                        sums[k] += float(f(sr_img, gt_img))
                n += 1
                if pbar is not None:
                    pbar.update(1)
        if pbar is not None:
            pbar.close()
        if metrics:
            results = {k: s / n for k, s in sums.items()}
            for k, v in results.items():
                self.update_best(name, k, v, current_iter)
            self.log_metrics(name, results, current_iter, tb_logger)

    def _img_path(self, vis: Path, dataset: str, img: str, current_iter) -> Path:
        if self.is_train:
            return vis / img / f"{img}_{current_iter}.png"
        suffix = (self.opt.get("val") or {}).get("suffix")
        suffix = self.opt["name"] if suffix is None else suffix
        return vis / dataset / f"{img}_{suffix}.png"

    # ---------------------------------------------------------------- checkpoint payloads
    def export_state_dicts(self) -> dict[str, dict[str, Tensor]]:
        """net_g: the EMA if enabled, else eval_weights() (SF: x); net_d: the live D."""
        with self.eval_weights() as net:
            out = {"net_g": cpu_copy(net.state_dict())}  # copied before the restore
        if self.net_d is not None:
            out["net_d"] = cpu_copy(self.net_d.state_dict())
        return out

    def run_record(self) -> dict[str, Any]:
        o = self.opt
        return {
            "network_g": o["network_g"],
            "network_d": o.get("network_d"),
            "use_amp": o.get("use_amp", False),
            "bfloat16": o.get("bfloat16", False),
            "ema": (o.get("train") or {}).get("ema", 0),
            "world_size": self.rt.world_size,
        }
