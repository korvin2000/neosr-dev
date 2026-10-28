"""Single-image super-resolution trainer (design T2 §3, §6, §8.4; T1 §4.5)."""

from collections import deque
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.ao.quantization import FakeQuantizeBase
from torch.ao.quantization.observer import ObserverBase
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm

from neosr.archs import build_network
from neosr.config import ConfigError, qat_schedule
from neosr.data.augmentations import apply_augment
from neosr.losses import build_losses
from neosr.losses.base import Context, restoring_clamp
from neosr.models.base import base, build_optimizer, build_schedulers
from neosr.models.checkpoint import cpu_copy, load_weights, read_weights
from neosr.models.phase import Ema, LossMeter, Phase, no_sync, trainable
from neosr.models.tiling import infer_tiled
from neosr.runtime import Runtime
from neosr.utils import get_root_logger, imwrite, tc, tensor2img
from neosr.utils.img_util import quantize

# optimizers without a native warmup_steps: the trainer prepends a LinearLR warmup (ST-10)
LINEAR_WARMUP = frozenset({"adam", "adamw", "nadam", "adan"})
QAT_KEYS = (
    "enable_qat", "qat_type", "fused_init", "quantize_residual", "calibration_iters",
    "qat_freeze_norm_iter", "qat_freeze_observers_iter",
)  # fmt: skip


def step_lr(group: dict[str, Any]) -> float:
    """The lr of the last step: adan_sf/adamw_sf scale `lr` by their own linear warmup (`step` and
    `k` count their applied steps); a trainer LinearLR warmup is already in `lr`."""
    w, n = group.get("warmup_steps") or 0, group.get("step", group.get("k", 0))
    return group["lr"] * min(1.0, n / w) if w else group["lr"]


class Buffers:
    """The buffers of a net (BN statistics and counters, spectral-norm u/v, QAT observers) and a
    preallocated copy taken at the start of every update window: a skipped update puts back what its
    forwards wrote (T-04). Like Ema, it assumes buffers keep their shape and change in place (QAT
    materialises its per-channel shapes in prepare_qat)."""

    def __init__(self, net: nn.Module) -> None:
        self.live = list(net.buffers())
        self.copy = [torch.empty_like(b) for b in self.live]

    @torch.no_grad()
    def save(self) -> None:
        if self.live:
            torch._foreach_copy_(self.copy, self.live)

    @torch.no_grad()
    def restore(self) -> None:
        if self.live:
            torch._foreach_copy_(self.live, self.copy)


@contextmanager
def observers_frozen(net: nn.Module) -> Generator[None]:
    """QAT: evaluation must not move activation ranges; eval() does not stop observers (T5 D11)."""
    fqs = [m for m in net.modules() if isinstance(m, FakeQuantizeBase)]
    flags = [m.observer_enabled.clone() for m in fqs]
    for m in fqs:
        m.disable_observer()
    try:
        yield
    finally:
        for m, f in zip(fqs, flags, strict=True):
            m.observer_enabled.copy_(f)


@contextmanager
def no_cudnn_autotune() -> Generator[None]:
    """benchmark off, nothing else: cudnn.flags() would also reset benchmark_limit and fp32_precision."""
    was = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark = was


def image_id(lq_path: str, root: str | None) -> Path:
    """The path of an image below its dataroot without suffix; the stem for LMDB keys or outside it."""
    p = Path(lq_path)
    try:
        return p.relative_to(root).with_suffix("") if root else Path(p.stem)
    except ValueError:
        return Path(p.stem)


class image(base):
    """Paired SR training and inference. `otf` overrides only how a training batch is made."""

    def __init__(self, opt: dict[str, Any], runtime: Runtime) -> None:
        super().__init__(opt, runtime)
        # model-side memory format follows AMP, as before; training inputs now match it (T4 P6)
        self.mf = torch.channels_last if opt.get("use_amp") else torch.contiguous_format
        resuming = self.is_train and opt["path"].get("resume_state") is not None
        self._qat_pretrain: tuple | None = None  # a QAT file waits for prepare_qat
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
        self.ema = Ema(self.net_g, ema, tr.get("ema_power")) if ema > 0 else None
        g_terms, d_terms = build_losses(
            tr, scale=self.scale, device=self.device, has_d=self.net_d is not None
        )
        self.accumulate = ds.get("accumulate", 1)
        # true: the legacy max-norm 1.0; a positive float; "spike"
        clip = tr.get("grad_clip", True)
        amp = {
            "amp_dtype": torch.bfloat16 if opt.get("bfloat16") else torch.float16,
            "use_amp": bool(opt.get("use_amp")),
            "accumulate": self.accumulate,
            "grad_clip": float(clip) if isinstance(clip, bool) else clip,
            "max_skip_streak": tr.get("max_skip_streak", 10),
            "strict_unused": self.rt.distributed
            and not opt.get("find_unused_parameters"),
        }
        self.phases = {"net_g": self._phase("g", self.net_g, g_terms, amp)}
        if self.net_d is not None:
            self.phases["net_d"] = self._phase("d", self.net_d, d_terms, amp)
        self.g, self.d = self.phases["net_g"], self.phases.get("net_d")
        self.meter = LossMeter()
        # "restore" (default, restoring_clamp); true: legacy hard [1/255, 1]; false: none
        self.clamp = tr.get("clamp", "restore")
        self.aug, self.aug_prob = ds.get("augmentation"), ds.get("aug_prob")
        self.aug_multi_prob = ds.get("aug_multi_prob", 0.3)
        self.aug_antialias = ds.get("aug_antialias", False)
        if self.aug is not None and list(self.aug) == ["none"]:
            self.aug = None

        if resuming:
            self.resume(Path(opt["path"]["resume_state"]))
        self.buffers = {n: Buffers(net) for n, net in self.nets.items()}
        # last: DDP broadcasts the loaded weights; compiled graphs see them in place
        self.net_g_fwd = self.wrap(self.net_g)
        # D: buffers synced once, not per forward (a batch-statistics D still runs 2 forwards
        # before 1 backward, where a broadcast in between would fail)
        self.net_d_fwd = self.wrap(self.net_d, broadcast_buffers=False)
        # G's adversarial pass runs D without a DDP reducer
        dist = self.rt.distributed
        self.net_d_adv = self.wrap(self.net_d, ddp=False) if dist else self.net_d_fwd

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
            strict = p.get(f"strict_load_{s}", True)
            qat_state = getattr(net, "qat_state", None) if key == "network_g" else None
            if qat_state is not None and (kw := qat_state(sd)) is not None:
                self._qat_pretrain = (sd, kw, strict)
            else:
                load_weights(net, sd, strict, log)
        self.log_network(net, key)
        return net

    def _prepare_qat(self) -> None:
        """T5 §2 (T2 §3 step 3): prepare QAT in place on the bare `self.net_g`, after the float/fused
        pretrain load and before anything captures parameter identity (EMA, optimizers, wrappers),
        then load a QAT-kind pretrain. Training sets the phase per update (`_qat_step`); inference
        evaluates the fake-quant net with frozen ranges, as convert.py exports it."""
        self.qat: tuple[int, int, int] | None = None
        self._qat_phase: tuple[bool, bool, bool] | None = None
        tr = (self.opt.get("train") or {}) if self.is_train else {}
        pre, self._qat_pretrain = self._qat_pretrain, None
        if not tr.get("enable_qat") and pre is None:
            return
        net, log = self.net_g, get_root_logger()
        if not hasattr(net, "prepare_qat"):
            name = self.opt["network_g"].get("type")
            msg = f"train.enable_qat: network_g {name!r} has no QAT support"
            raise ConfigError(msg)
        if self.is_train and not tr.get("enable_qat"):
            msg = "path.pretrain_network_g is a QAT checkpoint: set train.enable_qat = true to train it"
            raise ConfigError(msg)
        # the resolved paths: validate_train cannot know whether auto_resume finds a state
        p = self.opt["path"]
        loads = p.get("pretrain_network_g") or p.get("resume_state")
        if self.is_train and self.opt["network_g"].get("fused_init") and not loads:
            msg = (
                "network_g.fused_init=true builds the deploy topology (folded norms) and there is no "
                "pretrain_network_g or training state to load; QAT from scratch needs fused_init=false"
            )
            raise ConfigError(msg)
        kw = {k: tr[k] for k in ("qat_type", "quantize_residual") if k in tr}
        net.prepare_qat(**((pre[1] if pre else {}) | kw))
        if pre is not None:
            load_weights(net, pre[0], pre[2], log)
        if not self.is_train:
            net.set_qat_phase(fake_quant=True, observe=False, freeze_norm=True)
            return
        total = (self.opt.get("logger") or {}).get("total_iter", 1000000)
        self.qat = qat_schedule(tr, total)
        src = " from a QAT checkpoint" if pre else ""
        calib, norm_at, obs_at = self.qat
        log.info(
            f"QAT prepared{src}: observe-only through update {calib}, "
            f"norms frozen from {norm_at}, observers from {obs_at} (-1 = never)"
        )
        if self.opt.get("use_amp") or self.opt.get("compile"):  # both pass T5 A7
            log.warning(
                "QAT under use_amp/compile works but fp32 eager is the reference; compile gains "
                "little (dynamo does not trace torch.ao FakeQuantize: graph breaks, eager fallback)"
            )
        if self.rt.world_size > 1:
            log.info("QAT under DDP: observer and norm statistics follow rank 0")

    def _qat_step(self, it: int) -> None:
        """The phase of update `it` is a pure function of `it` (T5 D12/D13): resume- and DDP-safe."""
        calib, norm_at, obs_at = self.qat  # type: ignore[misc]
        phase = (it > calib, not 0 <= obs_at <= it, 0 <= norm_at <= it)
        if phase != self._qat_phase:
            self.net_g.set_qat_phase(*phase)  # type: ignore[operator]
            self._qat_phase = phase
            get_root_logger().info(
                f"QAT update {it}: fake_quant={phase[0]}, observers={phase[1]}, "
                f"norms_frozen={phase[2]}"
            )

    def _phase(
        self, name: str, net: nn.Module, terms: list, amp: dict[str, Any]
    ) -> Phase:
        section = f"train.optim_{name}"
        names, params = trainable(net)
        cfg = dict(self.opt["train"][f"optim_{name}"])
        warmup = 0
        if str(cfg.get("type")).lower() in LINEAR_WARMUP:
            warmup = cfg.pop("warmup_steps", 0)
            if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
                msg = (
                    f"[{section}] warmup_steps must be an integer >= 0, got {warmup!r}"
                )
                raise ConfigError(msg)
        kind, opt = build_optimizer(cfg, params, section)
        # built first: the configured scheduler then chains onto the warmed-up lr
        scheds = [LinearLR(opt, 1 / warmup, total_iters=warmup)] if warmup else []
        scheds += build_schedulers(self.opt["train"].get("scheduler"), opt)
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
                gt,
                lq,
                scale=self.scale,
                augs=self.aug,
                prob=self.aug_prob,
                multi_prob=self.aug_multi_prob,
                antialias=self.aug_antialias,
            )
        self.lq = lq.contiguous(memory_format=self.mf)
        self.gt = gt.contiguous(memory_format=self.mf)

    def optimize_parameters(self) -> bool:
        """One micro-batch. Returns True when it closed an update window (then `self.update` grew)."""
        if self.qat is not None:
            self._qat_step(self.update + 1)
        if self.micro % self.accumulate == 0:  # a window starts
            for b in self.buffers.values():
                b.save()
        boundary = (self.micro + 1) % self.accumulate == 0
        it = self.update + 1  # the update being computed
        g_on = self.g.active(it)
        if g_on or (self.d is not None and self.d.active(it)):
            # forward + losses + backward; an idle G only makes D's fake samples (no graph)
            with no_sync(self.net_g_fwd, boundary), torch.set_grad_enabled(g_on):
                with torch.autocast(**self.g.autocast):
                    sr = self.net_g_fwd(self.lq)
                    if self.clamp == "restore":
                        sr = restoring_clamp(sr)
                    elif self.clamp:
                        sr = sr.clamp(1 / 255, 1)
                ctx = Context(
                    sr=sr,
                    gt=self.gt,
                    lq=self.lq,
                    it=it,
                    net_d=self.net_d_fwd,
                    net_d_adv=self.net_d_adv,
                )
                self.g.backward(ctx)
            g_logs = dict(ctx.logs)
            self.meter.add(g_logs, "g")
            if self.d is not None:
                with no_sync(self.net_d_fwd, boundary):
                    self.d.backward(ctx)
                self.meter.add(
                    {k: v for k, v in ctx.logs.items() if k not in g_logs}, "d"
                )
        else:  # every phase idle: no forward (compute, RNG, buffer writes); the clocks still run
            for p in self.phases.values():
                p.idle = True
        self.micro += 1
        if not boundary:
            return False
        # True applied, False skipped (its buffers are put back), None idle
        ok = {self.g: self.g.step()}
        if ok[self.g] is False:
            self.buffers["net_g"].restore()
        if ok[self.g] and self.ema is not None:
            self.ema.update()
        elif self.qat is not None and self.ema is not None and self.ema.lists is None:
            # QAT: until the first successful update copies the net, the EMA mirrors its observed
            # ranges and phase flags; the +-inf observer reset would fail check_finite (T5 D11)
            with torch.no_grad():
                buffers = [*self.ema.module.buffers()], [*self.net_g.buffers()]
                torch._foreach_copy_(*buffers)
        if self.d is not None:
            ok[self.d] = self.d.step()
            if ok[self.d] is False:
                self.buffers["net_d"].restore()
        # means over applied steps: a skipped or idle one adds nothing
        for p, s in ok.items():
            if s:
                clipped = {} if p.clipped is None else {f"clipped_{p.name}": p.clipped}
                self.meter.add({f"grad_norm_{p.name}": p.grad_norm, **clipped}, p.name)
        # each phase's values by its own step: an applied G with a broken loss raises (T1 D15)
        # even when D skipped
        self.meter.commit({p.name: s is not False for p, s in ok.items()})
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
        ts, ranges = [], []
        for n in nets:
            obs = {
                id(b)
                for m in n.modules()
                if isinstance(m, ObserverBase)
                for b in m.buffers()
            }
            for t in (*n.parameters(), *n.buffers()):
                if t.is_floating_point():
                    (ranges if id(t) in obs else ts).append(t)
        bad = ~torch.stack(torch._foreach_norm(ts)).isfinite().all()
        if ranges:  # QAT observer min/max: +-inf is their reset (nothing observed yet), NaN poison
            bad |= torch.stack(torch._foreach_norm(ranges)).isnan().any()
        bad = bad.int()
        # buffers are rank-local: one bad rank stops every rank
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(bad, op=torch.distributed.ReduceOp.MAX)
        if bool(bad):
            msg = "non-finite weights or buffers; the last committed checkpoint stands"
            raise FloatingPointError(msg)

    def current_lr(self) -> list[float]:
        return [step_lr(g) for g in self.g.opt.param_groups]

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
                params = list(net.parameters())
                snap = [torch.empty_like(p) for p in params]
                with torch.no_grad():
                    torch._foreach_copy_(snap, params)
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

    def _infer(self, net: nn.Module, lq: Tensor, warned: list[bool]) -> Tensor:
        """Full image, or tiled when val.tile > 0. A CUDA OOM retries the image once tiled (val.tile or
        256): the caching allocator is emptied first; a second OOM propagates (T-03)."""
        try:
            if self.tile <= 0:
                return net(lq)
            return infer_tiled(net, lq, self.scale, self.tile, self.tile_overlap)
        except torch.OutOfMemoryError:
            pass  # retried outside the handler: the traceback no longer pins the failed activations
        torch.cuda.empty_cache()
        tile = self.tile if self.tile > 0 else 256
        if not warned[0]:
            warned[0] = True
            get_root_logger().warning(
                f"validation ran out of memory on a {tuple(lq.shape[-2:])} input: retrying tiled "
                f"(tile {tile}, tile_overlap {self.tile_overlap}); tiled metrics can differ "
                "slightly from full-image ones. Set val.tile to avoid this."
            )
        return infer_tiled(net, lq, self.scale, tile, self.tile_overlap)

    def validate(self, loader, current_iter, tb_logger, save_img: bool) -> None:
        """Output contract of the old nondist_validation (T2 §6.3); rank 0, called by validation().
        Metrics run on the quantised images (the values tensor2img writes, MT-04) where the output is:
        the device, or the CPU for a tiled one (its full-size fp64 SSIM statistics); a
        uint8 host copy is made only for an image that is saved or logged, PNGs are written by one
        background thread. Inference (test.py) honours use_amp/bfloat16; training validation stays
        fp32 (GP-13). cuDNN autotuning is off: every val shape would be timed anew (MT-11)."""
        ds, val = loader.dataset.opt, self.opt.get("val") or {}
        name, vis = ds["name"], Path(self.opt["path"]["visualization"])
        root = ds.get("dataroot_lq")
        metrics = {} if ds["type"] == "single" else self.metrics
        values: dict[str, list[Tensor]] = {k: [] for k in metrics}
        pbar = (
            tqdm(total=len(loader), unit="image", colour="green", ascii=" >=")
            if val.get("pbar", True)
            else None
        )
        save_tb = (
            self.is_train
            and tb_logger is not None
            and (self.opt.get("logger") or {}).get("save_tb_img", False)
        )
        save_lq = save_img and val.get("save_lq", True)
        amp = not self.is_train and bool(self.opt.get("use_amp"))
        autocast = torch.autocast(
            self.device.type,
            dtype=torch.bfloat16 if self.opt.get("bfloat16") else torch.float16,
            enabled=amp and self.device.type == "cuda",
        )
        warned = [False]
        writer, pending = (
            ThreadPoolExecutor(1, thread_name_prefix="val_imwrite"),
            deque(),
        )

        def write(img, path: Path) -> None:
            pending.append(writer.submit(imwrite, img, str(path)))
            while len(pending) > 2:  # bounded: at most two images wait in memory
                pending.popleft().result()

        # one swap per validation; QAT ranges stay put
        try:
            with self.eval_weights() as net, observers_frozen(net), no_cudnn_autotune():
                for data in loader:
                    rel = image_id(data["lq_path"][0], root)
                    if pbar is not None:  # the image being inferred, not the last one
                        pbar.set_description(
                            f"{tc.light_green}Inferring on {rel}{tc.end}"
                        )
                    with torch.inference_mode():
                        lq = data["lq"].to(self.device, non_blocking=True)
                        with autocast:
                            sr = self._infer(net, lq, warned)
                        if metrics or save_img or save_tb:  # else nothing consumes sr
                            q = quantize(sr)  # where sr is (tiled: the CPU)
                        del sr
                        if metrics:  # where q is: a tiled image stays on the CPU, whose
                            # fp64 SSIM statistics would not fit the device at that size
                            gt = quantize(data["gt"].to(q.device, non_blocking=True))
                            for k, f in metrics.items():
                                values[k].append(
                                    f(q, gt).to(self.device, non_blocking=True)
                                )
                            del gt
                    if save_img or save_tb:  # the one host copy: uint8 HWC RGB
                        rgb = q[0].to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                        if save_img:  # tensor2img's layout: BGR, or HW for gray
                            bgr = rgb[..., 0] if rgb.shape[-1] == 1 else rgb[..., ::-1]
                            path = self._img_path(vis, name, rel, current_iter)
                            write(np.ascontiguousarray(bgr), path)
                            lq_path = self._lq_path(vis, name, rel)
                            if save_lq and not lq_path.exists():
                                write(tensor2img([data["lq"]]), lq_path)
                        if save_tb:
                            tb_logger.add_image(
                                f"{name}/{rel.as_posix()}",
                                rgb,
                                global_step=current_iter,
                                dataformats="HWC",
                            )
                    q = lq = None  # nothing of this image lives into the next forward
                    if pbar is not None:
                        pbar.update(1)
        finally:
            writer.shutdown(wait=True)
            if pbar is not None:
                pbar.close()
        for f in pending:  # a failed write raises here
            f.result()
        if metrics:
            # one sync for the event; summed in Python in image order, as before
            rows = torch.stack([torch.stack(v).double() for v in values.values()])
            results = {
                k: sum(row) / len(row)
                for k, row in zip(values, rows.tolist(), strict=True)
            }
            for k, v in results.items():
                self.update_best(name, k, v, current_iter)
            self.log_metrics(name, results, current_iter, tb_logger)

    def _img_path(self, vis: Path, dataset: str, rel: Path, tag) -> Path:
        """`rel`: the image's path below its dataroot without suffix (image_id); dataset and
        subfolders keep same-named images apart (F21). Training: one folder per image,
        vis/<dataset>/<rel>/<stem>_<iter>.png. Test: the outputs alone in
        vis/<dataset>/<rel parent>/<stem>_<suffix>.png."""
        if self.is_train:
            return vis / dataset / rel / f"{rel.name}_{tag}.png"
        suffix = (self.opt.get("val") or {}).get("suffix")
        suffix = self.opt["name"] if suffix is None else suffix
        return vis / dataset / rel.parent / f"{rel.name}_{suffix}.png"

    def _lq_path(self, vis: Path, dataset: str, rel: Path) -> Path:
        """Training: vis/<dataset>/<rel>/<stem>_lq.png next to the outputs. Test:
        vis/lq/<dataset>/<rel>.png, apart from the outputs whatever the run is called."""
        if self.is_train:
            return vis / dataset / rel / f"{rel.name}_lq.png"
        return vis / "lq" / dataset / f"{rel}.png"

    # ---------------------------------------------------------------- checkpoint payloads
    def export_state_dicts(self) -> dict[str, dict[str, Tensor]]:
        """net_g: the EMA if enabled, else eval_weights() (SF: x); net_d: the live D."""
        with self.eval_weights() as net:
            out = {"net_g": cpu_copy(net.state_dict())}  # copied before the restore
        if self.net_d is not None:
            out["net_d"] = cpu_copy(self.net_d.state_dict())
        return out

    def run_record(self) -> dict[str, Any]:
        """What a resume compares (each change warns): nets, precision, objective, optimisation."""
        o, tr, ds = self.opt, self.opt["train"], self.opt["datasets"]["train"]
        return {
            "network_g": o["network_g"],
            "network_d": o.get("network_d"),
            "use_amp": o.get("use_amp", False),
            "bfloat16": o.get("bfloat16", False),
            "ema": tr.get("ema", 0),
            "world_size": self.rt.world_size,
            "optim_g": tr["optim_g"],
            "optim_d": tr.get("optim_d"),
            "scheduler": tr.get("scheduler"),
            "losses": {k: v for k, v in tr.items() if k.endswith("_opt")},
            "train": {
                "clamp": self.clamp,
                "grad_clip": tr.get("grad_clip", True),
                "ema_power": tr.get("ema_power"),
                **{k: tr.get(k) for k in QAT_KEYS},
            },
            "data": {
                "accumulate": self.accumulate,
                "batch_size": ds.get("batch_size"),
                "patch_size": ds.get("patch_size"),
            },
        }
