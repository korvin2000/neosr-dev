# T5 — AetherNet registration + end-to-end QAT (synthesized design)

Scope: make `options/aether/*.toml` run (`aether_small|medium|large`) and make QAT correct from build to INT8 QDQ ONNX.
Synthesized from the Fable and Codex proposals. Opus re-checked the disputed points against the code and on the GPU
(RTX 4090, torch 2.9.1+cu128). The probe is in `scratchpad/design/qat/synth/probe.py`; no repository edits were made.

## 0. Opus verification of the disputed points (new evidence)

| Probe (aether_small x2, built from scratch, 14 steps, EMA 0.999) | fp32 | bf16 autocast | fp16 autocast + GradScaler |
|---|---|---|---|
| FakeQuantize / QAT modules after the synthesized prepare | 226 / 71 | 226 / 71 | 226 / 71 |
| Finite loss; EMA `update_parameters` from step 1 (EMA created after priming) | ok | ok | ok |
| DeploymentNorm running stats change before the freeze_norm step, constant after it | ok | ok | ok |
| Activation scales change before the observer freeze, constant after it; weights keep updating | ok | ok | ok |
| Save (EMA, `module.` stripped like `base.py:347-349`) → fresh build → `prepare_qat` → strict load | diff 0 | diff 0 | diff 0 |
| Folding the norms on an eval-mode export copy | diff 0 | diff 0 | diff 0 |
| ONNX (dynamo=False, opset 17, scalarized Q/DQ): Conv nodes with input+weight DQ | 71/74 | 71/74 | 71/74 |
| torch fake-quant vs ORT `ORT_DISABLE_ALL` (40x32) | max 1.9e-3, mean 5e-6 | — | — |
| torch fake-quant vs ORT default optimizations | max 5e-3, mean 6e-4 | 5e-3 | 5.4e-3 |

Consequences: (a) the Q/DQ export does not need the norms folded before QAT, so QAT **from scratch** works, which the
shipped tomls require (`options/aether/*_l1pretrain*.toml:41-42`: `enable_qat = true`, no `pretrain_network_g`).
(b) Plain `FakeQuantize` works under CUDA autocast in both fp16 and bf16. Codex's failure was the *fused* FQ on CPU.
(c) `convert.py:assert_verify` (`rtol=.01, atol=1e-3`, `convert.py:155-157`) is too tight for optimized ORT on a
QDQ model. Rounding at quantization ties flips single LSBs, so QAT models need their own tolerance (§5, S4).

## 1. Decisions table

"Fable / Codex" show each proposal. "Final" is the synthesized decision: = means all three agree, R means Opus resolved
a disagreement with the evidence cited.

| # | Topic | Fable | Codex | Final |
|---|---|---|---|---|
| D1 | Canonical source | `neosr/archs/aether_arch.py` | same | = keep `aether_arch.py` (byte-identical to `common/v4`; `*_core.py` are different archs, never imported: scanner loads `*_arch.py` only, `neosr/archs/__init__.py:18-27`) |
| D2 | Delete | `aether_core.py`, `common/{v4,v5,aether_scripts}`, train.py synthetic calib + QAT block + sys.path hack, `_init_ema`, arch `calibrate/convert_to_quantized/export_onnx`, warnings filter | same set (after consolidating scripts) | = same set. `common/v5/Readme.md` deployment notes move to the wiki/docs. Out of scope: `image.py.helaman` (not aether) |
| D3 | Registry | class `aether` + 3 decorated functions, `**kw` | 3 functions, fixed width/depth/attn | = `@ARCH_REGISTRY.register()` on class `aether` and `aether_small/medium/large(scale, **kw)`. Passing `embed_dim/depths/use_*_attn` to a named variant raises `TypeError` (duplicate kwarg), which is the reject Codex asks for |
| D4 | Scale | transitional `net_opt()` default (span pattern) | injected by `build_network` | R: `scale` is a factory kwarg. Default comes from `net_opt()` only until the build_network injection lands, then the default is removed. The ctor validates `scale in (2,3,4,8)` (scale 1 silently builds 2x: `aether_arch.py:341-347`) |
| D5 | Fusion before QAT | `fuse_model()` = reparam + **fold norms** (so QAT needs a float ckpt) | fuse **reparam only**, keep DeploymentNorm live, fold at export | R → Codex. Supports from-scratch QAT (the tomls) and fine-tunes with one path. Folding at export is exact (§0). `[train].fused_init=false` (separately fake-quantized lk/sk branches) is rejected in v1 |
| D6 | Q/DQ boundaries | add `QuantStub`s after conv_first / norms / SiLU (71/74 eligible) | `QuantWrapper` per conv + `ResidualAdd`, remove `is_quantized` | R → Fable (verified ORT parity; fewer state-dict key changes; keeps existing `FloatFunctional` paths). The stubs are **new attributes**, not wrappers, so float checkpoint keys stay unchanged. The QuantWrapper design remains the fallback if native CPU INT8 is ever required |
| D7 | QConfig | symmetric qint8 act [-128,127], per-channel symmetric qint8 weights (TensorRT/ORT) | `get_default_qat_qconfig("fbgemm", version=0)` | R → Fable. Target is QDQ ONNX (tomls: "int8 onnx"). TRT rejects non-zero zero points; fbgemm quint8 asymmetric would not serve TRT. Plain `FakeQuantize` (not fused) for ONNX symbolics and autocast |
| D8 | Sensitive layers | `qconfig=None` before prepare | same | = `quant`, `conv_first`, `conv_before_upsample.0`, `conv_last` get `qconfig=None` **before** `tq.prepare_qat` (today `disable_*` after prepare leaves QAT modules, `aether_arch.py:560-565`) |
| D9 | Where to prepare | `image.__init__`, before `model_to_device` | same, before optimizer/EMA/DDP | = prepare in `image.__init__`: build → (load float) → `prepare_qat` → (load qat) → `model_to_device` (channels_last/compile/DDP, `base.py:147-176`) → `init_training_settings` (optimizer, EMA) |
| D10 | Per-channel buffer shapes for EMA | dummy forward in prepare (train mode) | prime with real batches before EMA | R: dummy forward in prepare in **eval** mode. Train mode would push random-input stats into DeploymentNorm running stats (`aether_arch.py:225-228`) of a loaded float ckpt. Then reset the per-tensor act observers in place with `fill_(±inf)` |
| D11 | EMA | keep `AveragedModel(use_buffers=False)`; buffers copied (`swa_utils.py:307-311`) | same; disable EMA observers for validation | = keep it; build from `self.net_g_bare` (compile `_orig_mod.` keys / DDP wrapper otherwise end up in the EMA). Before every validation forward: `model.apply(tq.disable_observer)` (`eval()` does not stop observers; buffer copy re-enables them at the next update) |
| D12 | Calibration data | real batches, in-loop: first `calibration_iters` optimizer steps observe-only (FQ off) | real batches, separate no-grad warmup before training | R → Fable: in-loop and computed from the step, so it is resume-safe, DDP-safe and needs no extra loader. Synthetic `generate_calib_data` deleted |
| D13 | Freeze schedule | observers at `total_iter//2` (from `image.total_iter`, wrong source) | norms at 70%, observers at 80% of the QAT updates | R → Codex. New optional keys `qat_freeze_norm_iter` (default `int(.7*T)`) and `qat_freeze_observers_iter` (default `int(.8*T)`), `-1` = never. T = `[logger].total_iter` (optimizer steps, `train.py:154`), **not** `image.total_iter` (`[train].total_iter`, default 200000, `image.py:159`). Norm freeze = `DeploymentNorm.eval()`, re-asserted after every `.train()` |
| D14 | AMP | allowed (verified fp16/bf16) | reject | R: allowed with a warning (verified in §0); the tomls keep `use_amp=false` |
| D15 | compile | allowed (verified) | reject in v1 | R: allowed with a warning; must pass acceptance A7, otherwise turned into an error |
| D16 | SAM | reject | — | = reject (`closure` runs twice through observers; `sam_optimizer_g` not in `self.optimizers`) |
| D17 | DDP | prepare before wrap; rank-0 buffer broadcast accepted | same; explicit rank-0 policy | = rely on the DDP default `broadcast_buffers=True` (rank-0 observer/norm stats every forward). The phase is a pure function of the step, so it is identical on all ranks |
| D18 | Checkpoint format | existing `{"params": ema_sd}`, kind inferred from keys | metadata + raw + full EMA + n_averaged | R → Fable (smallest; fits `save_network`). Codex's "resume loads EMA weights into net_g" (`image.py:990-995`, `misc.py:157-158`) is pre-existing and arch-independent, so it is a known issue here, not fixed |
| D19 | `fused` flag persistence | make `DeploymentNorm.fused` a buffer | split reparam vs norm folding | = bool buffer `fused_flag` + load post-hook that syncs the python `fused`. The ctor passes `fused=fused_init` to DeploymentNorm. `fuse_model()` drops the early return (`aether_arch.py:519-521`) |
| D20 | Deployment artifact | QDQ ONNX from the fake-quant model; eager INT8 dropped (verified failure) | native CPU INT8 + QDQ ONNX + float ONNX | R → Fable: QDQ ONNX (+ `--strip-qat` float fused ckpt). Native CPU INT8 is out of v1 (would need the D6 fallback) |
| D21 | ONNX export details | `dynamo=False`, opset 17, scalarize 1-elem Q/DQ scale/zp, observers off + FQ on, distinct out dim names | same route (source-checked) | = same; QAT export runs on a CPU deep copy; opset ≥ 13 required |
| D22 | Ctor hygiene | assert `embed_dim % len(depths) == 0` | + mean buffer for any `in_chans`, init fused models | = add both: `mean = torch.full((1,in_chans,1,1), .5)`; always `_init_weights` (the `fused_init` print-and-skip at `:471-474` removed) |

## 2. Lifecycle (normative)

```
image.__init__ (is_train and [train].enable_qat):
  mirror [train].quantize_residual -> [network_g] (conflict -> exit) ; validate (§4)
  net = build_network(network_g)                       # CPU, float topology
  kind = ckpt_kind(pretrain_network_g) | "scratch"     # qat | fused | float | scratch
  fused  -> net.fuse_model() ; float/fused -> load_network(strict)   # load BEFORE reparam fusion uses weights
  net.prepare_qat(train_opt)                           # fuse_reparam, qconfig, prepare_qat, prime (eval), reset act observers
  qat    -> load_network(strict)                       # resume: AFTER topology exists (FQ _load_from_state_dict resizes)
  self.net_g_bare = net ; self.net_g = model_to_device(net)  # channels_last / compile / DDP
  init_training_settings(): optimizer from net_g params (FQ adds buffers only), AveragedModel(net_g_bare)
optimize_parameters(current_iter): first line -> _qat_schedule(current_iter)
  step = (current_iter-1)//accum_iters + 1
  fake_quant = step > calibration_iters ; observe = not(0 <= freeze_obs <= step) ; freeze_norm = 0 <= freeze_norm_iter <= step
nondist_validation: disable observers on the model used -> infer -> model.train() -> re-apply cached phase to net_g_bare
save: unchanged (EMA module state incl. FQ buffers) ; resume: kind=qat path above + existing optimizer/scheduler resume
convert.py: kind-aware build/load -> eval -> phase(FQ on, observers off, norms frozen) -> fold_norms -> CPU -> ONNX(dynamo=False) -> scalarize Q/DQ -> check -> ORT parity
```

## 3. Code skeleton (115 lines)

```python
# ===== neosr/archs/aether_arch.py (replaces :26-29, :533-594, :596-642, :644-667; small ctor/forward edits) =====
from neosr.archs.arch_util import net_opt                  # transitional (D4)
from neosr.utils.registry import ARCH_REGISTRY
upscale, __ = net_opt()
FLOAT_LAYERS = ("quant", "conv_first", "conv_before_upsample.0", "conv_last")
QAT_TAGS = ("fake_quant", "activation_post_process")
def ckpt_kind(sd: dict) -> str:
    if any(t in k for k in sd for t in QAT_TAGS): return "qat"
    return "fused" if any(k.endswith("fused_conv.weight") for k in sd) else "float"
def qat_qconfig(qat_type: str = "int8") -> tq.QConfig:    # TensorRT/ORT policy (D7)
    if qat_type != "int8": raise ValueError(f"qat_type={qat_type!r}; supported: 'int8'")
    a = tq.FakeQuantize.with_args(observer=tq.MovingAverageMinMaxObserver, quant_min=-128, quant_max=127,
                                  dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    w = tq.FakeQuantize.with_args(observer=tq.MovingAveragePerChannelMinMaxObserver, quant_min=-128, quant_max=127,
                                  dtype=torch.qint8, qscheme=torch.per_channel_symmetric, ch_axis=0)
    return tq.QConfig(activation=a, weight=w)
# DeploymentNorm.__init__(channels, fused=False):  (D19)
#     self.fused = fused; self.register_buffer("fused_flag", torch.tensor(fused))
#     self.register_load_state_dict_post_hook(lambda mod, _: setattr(mod, "fused", bool(mod.fused_flag)))
#   fuse(): ... existing ...; self.fused = True; self.fused_flag.fill_(True)
# AetherBlock: self.norm = DeploymentNorm(dim, fused=fused_init); self.quant_norm = tq.QuantStub()
#   forward: x = self.quant_norm(self.norm(x))           # QuantStub is identity until prepared
# GatedConvFFN: self.quant_act = tq.QuantStub(); forward: activated = self.quant_act(self.act(gate))
# aether.__init__: if scale not in (2, 3, 4, 8): raise ValueError; assert embed_dim % len(depths) == 0
#   mean = torch.full((1, in_chans, 1, 1), 0.5); self.norm = DeploymentNorm(embed_dim, fused=fused_init)
#   self.quant_first = tq.QuantStub(); self.quant_body = tq.QuantStub(); always self.apply(self._init_weights)
#   forward: x = self.quant_first(self.conv_first(x)) ... x = self.conv_after_body(self.quant_body(self.norm(x))) + x
class aether(nn.Module):  # new/replaced methods only
    def fuse_reparam(self) -> None: [m.fuse() for m in self.modules() if isinstance(m, ReparamLargeKernelConv)]
    def fold_norms(self) -> None: [m.fuse() for m in self.modules() if isinstance(m, DeploymentNorm)]
    def fuse_model(self) -> None:                          # deploy topology; idempotent, no early return
        self.fuse_reparam(); self.fold_norms(); self.fused_init = True
    def prepare_qat(self, train_opt: dict) -> None:
        if self.is_quantized: return
        self.fuse_reparam()                                # norms stay live (D5)
        self.qconfig = qat_qconfig(train_opt.get("qat_type", "int8"))
        for name in FLOAT_LAYERS: self.get_submodule(name).qconfig = None
        tq.prepare_qat(self.train(), inplace=True)
        for mod in self.modules():
            if hasattr(mod, "is_quantized"): mod.is_quantized = True
        self.eval()                                        # D10: materialize per-channel scale shapes, no norm-stat update
        with torch.no_grad(): self(torch.rand(1, self.mean.shape[1], 16, 16, device=self.mean.device))
        for mod in self.modules():
            if isinstance(mod, tq.FakeQuantize) and not mod.is_per_channel:
                mod.activation_post_process.min_val.fill_(float("inf"))
                mod.activation_post_process.max_val.fill_(float("-inf"))
        self.train()
    def set_qat_phase(self, fake_quant: bool, observe: bool, freeze_norm: bool) -> None:
        self.apply(tq.enable_fake_quant if fake_quant else tq.disable_fake_quant)
        self.apply(tq.enable_observer if observe else tq.disable_observer)
        for mod in self.modules():
            if isinstance(mod, DeploymentNorm): mod.train(self.training and not freeze_norm)
    def float_state_dict(self) -> dict:                    # --strip-qat; loads strict with [network_g].fused_init=true
        net = copy.deepcopy(self).eval(); net.fold_norms()
        return {k: v for k, v in net.state_dict().items() if not any(t in k for t in QAT_TAGS)}
ARCH_REGISTRY.register()(aether)
@ARCH_REGISTRY.register()
def aether_small(scale: int = upscale, **kw): return aether(embed_dim=96, depths=(4, 4, 4, 4), scale=scale, use_channel_attn=False, use_spatial_attn=False, **kw)
@ARCH_REGISTRY.register()
def aether_medium(scale: int = upscale, **kw): return aether(embed_dim=128, depths=(6, 6, 6, 6), scale=scale, use_channel_attn=True, use_spatial_attn=True, **kw)
@ARCH_REGISTRY.register()
def aether_large(scale: int = upscale, **kw): return aether(embed_dim=180, depths=(8, 8, 8, 8, 8), scale=scale, use_channel_attn=True, use_spatial_attn=True, **kw)
# ===== neosr/models/image.py (replaces :35-58 net_g part; deletes :74-86; EMA :98-99 and :768 use net_g_bare) =====
    def __init__(self, opt):
        super().__init__(opt)
        tr = opt.get("train") or {}
        self.qat = bool(self.is_train and tr.get("enable_qat", False))
        if self.qat: self._qat_mirror_and_validate(opt)    # §4; logger.error + sys.exit(1) like :371-377
        net = build_network(opt["network_g"])
        load_path = opt["path"].get("pretrain_network_g")
        if self.qat:
            kind = ckpt_kind(self._peek_params(load_path)) if load_path else "scratch"   # CPU torch.load, same key pick as base.py:413-427
            if kind == "fused": net.fuse_model()
            if kind in ("float", "fused"): self.load_network(net, load_path, None, True)
            net.prepare_qat(tr)                            # BEFORE device/compile/DDP/EMA/optimizer
            if kind == "qat": self.load_network(net, load_path, None, True)
            get_root_logger().info(f"QAT prepared from {kind} weights.")
            load_path = None
        self.net_g_bare = net
        self.net_g = self.model_to_device(net)
        # ... net_d unchanged; `if load_path is not None:` existing pretrain_g load (non-QAT) unchanged ...
    def _qat_schedule(self, current_iter: int) -> None:   # first statement of optimize_parameters()
        step = (current_iter - 1) // self.accum_iters + 1  # optimizer-step units (train.py:565)
        tr, total = self.opt["train"], self.opt["logger"]["total_iter"]
        fn = tr.get("qat_freeze_norm_iter", int(0.7 * total)); fo = tr.get("qat_freeze_observers_iter", int(0.8 * total))
        phase = (step > tr.get("calibration_iters", 0), not (0 <= fo <= step), 0 <= fn <= step)
        if phase != getattr(self, "_qat_phase", None):
            self.net_g_bare.set_qat_phase(*phase); self._qat_phase = phase
            get_root_logger().info(f"QAT step {step}: fake_quant={phase[0]} observers={phase[1]} norms_frozen={phase[2]}")
    # nondist_validation (:891-899): `if self.qat: model.apply(tq.disable_observer)` before inference;
    #   after `model.train()`: `if self.qat and hasattr(self, "_qat_phase"): self.net_g_bare.set_qat_phase(*self._qat_phase)`
# ===== convert.py: load_net() (:76-129) and to_onnx() (:201) =====
    net_opt = {"type": args.network, "scale": args.scale}  # until build_network injects scale
    net = build_network(net_opt)
    kind = ckpt_kind(load_net) if hasattr(net, "prepare_qat") else None
    if kind == "fused": net.fuse_model()
    if kind == "qat": net.prepare_qat({"qat_type": "int8"})
    net.load_state_dict(load_net, strict=True)
    if kind is not None:
        net.eval()
        if kind == "qat":                                   # observers OFF (aten::copy), FQ ON (else no Q/DQ)
            net.set_qat_phase(fake_quant=True, observe=False, freeze_norm=True); net.fold_norms()
        else:
            net.fuse_model()                                # float/fused -> deploy topology, plain fp32 ONNX
        net = net.cpu()                                     # aether export on CPU (dummy input is CPU)
    torch.onnx.export(net, dummy_input, output_fp32, dynamo=False, opset_version=args.opset, do_constant_folding=True,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch", 2: "h", 3: "w"}, "output": {0: "batch", 2: "h_out", 3: "w_out"}})
    mdl = onnx.load(output_fp32); inits = {i.name: i for i in mdl.graph.initializer}
    for node in mdl.graph.node:                             # Q/DQ per-tensor scale/zero_point must be 0-d (ORT QLinearAdd)
        if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
            for name in node.input[1:3]:
                if name in inits and list(inits[name].dims) == [1]: del inits[name].dims[:]
    onnx.checker.check_model(mdl); onnx.save(mdl, output_fp32)
    # --strip-qat: torch.save({"params": net.float_state_dict()}, out)  (before the export branch)
```

## 4. Config keys and validation (`[train]`, checked in `image.__init__` before build)

| Key | Type / default | Contract |
|---|---|---|
| `enable_qat` | bool, false | net must have `prepare_qat`; else exit. QAT from scratch allowed; a float/fused/qat `pretrain_network_g` is detected by keys |
| `qat_type` | str, `"int8"` | only `"int8"` (symmetric TRT/ORT policy); else exit |
| `fused_init` | bool, true | `true` = reparam convs fused before QAT. `false` → exit (unsupported in v1). Distinct from `[network_g].fused_init` = ctor flag for loading **deployed** (reparam+norm fused) ckpts (`options/test_aether.toml:20`) |
| `quantize_residual` | bool, arch default true | mirrored into `[network_g]` if absent; both set and different → exit (`aether_arch.py:273-274`, v3 toml uses false) |
| `calibration_iters` | int ≥ 0 (not bool), 0 | first N optimizer steps: observers on, fake quant off |
| `qat_freeze_norm_iter` (new) | int, `int(.7*[logger].total_iter)`, -1 = never | DeploymentNorm → eval from this step |
| `qat_freeze_observers_iter` (new) | int, `int(.8*…)`, -1 = never | observers off from this step; FQ stays on |

Cross-checks: `calibration_iters < qat_freeze_norm_iter <= qat_freeze_observers_iter` when enabled; `train.sam` → exit;
`use_amp` / `compile` → warning; `world_size>1` → info line about rank-0 observer stats; `[network_g].fused_init=true` with
`enable_qat` and no pretrain → exit (deploy topology has folded, non-normalizing norms).

## 5. File ownership and steps

| Step | Files | Change | Owner tier |
|---|---|---|---|
| S1 | `neosr/archs/aether_arch.py` | D3, D4, D19, D22, stubs (D6), `fuse_reparam/fold_norms/fuse_model/prepare_qat/set_qat_phase/float_state_dict`, `ckpt_kind/qat_qconfig`; delete `calibrate/convert_to_quantized/export_onnx`, lambdas, warnings filter | t3 |
| S2 | `neosr/models/image.py` | QAT block in `__init__`, `_qat_mirror_and_validate`, `_peek_params`, `_qat_schedule`, validation guard, EMA on `net_g_bare`; delete `_init_ema` | t3 |
| S3 | `train.py` | delete `generate_calib_data` (:241-268), QAT block (:414-446), `common` sys.path hack (:665-679 region) | t2 |
| S4 | `convert.py` | scale in `net_opt`, aether/QAT branch, `dynamo=False`, distinct output dims, Q/DQ scalarization, `--strip-qat`, QAT tolerance in `assert_verify` (ORT_DISABLE_ALL: mean ≤1e-4, max ≤5e-3; default opt: mean ≤2e-3) | t2 |
| S5 | delete | `neosr/archs/aether_core.py`, `common/` (v4, v5, aether_scripts) | t1 |
| S6 | `options/aether/*.toml`, `options/test_aether.toml` | no key changes needed; optionally document the new freeze keys; test toml must point at a `--strip-qat` ckpt of the canonical arch (current path is an aether_core artifact) | t1 |

Dependencies: S1 → S2/S4; S3 after S2; S5 last (grep for `aether_core|from common` must be empty).
If the build_network scale-injection task lands first, S1 omits `net_opt` and S4 omits the explicit `"scale"`.

## 6. Acceptance checks (tiny GPU smoke, ≈5 min)

Synthetic paired set of 16 images in scratch. Config `debug_aether_qat` (debug → val/save freq 8, `options.py:267-271`):
`aether_small`, scale 2, patch 32, batch 4, `[logger].total_iter=32`, `ema=0.999`.

| ID | Check | Pass criterion |
|---|---|---|
| A1 | Registry | `build_network({"type": n, "scale": s})` for the 3 names × s∈{2,3,4}: output = s×input; scale 1 raises; `aether_small(embed_dim=64)` raises TypeError |
| A2 | Float run (`enable_qat=false`, 16 it) | trains, saves `net_g_16.pth`; `ckpt_kind == "float"` |
| A3 | QAT from scratch (shipped toml shape: no pretrain), `calibration_iters=4`, freeze_norm 12, freeze_obs 20 | log shows phases at steps 1/5/12/20; no EMA error at step 1; 71 qat modules / ≥226 FakeQuantize (x2); saved ckpt `ckpt_kind == "qat"` |
| A4 | QAT fine-tune from `net_g_16.pth` | strict float load before prepare; norm running stats unchanged by prepare (priming in eval) |
| A5 | `--auto_resume` from the A3 ckpt to 32 it | strict load after prepare; phase applied immediately = (True, True/False per step, …); optimizer state loads; EMA output of loaded net == saved EMA output (bit-exact) |
| A6 | Validation | observer buffers of the validated model unchanged across a validation pass; after it, DeploymentNorms of net_g are in eval iff past freeze_norm |
| A7 | Variants | `use_amp=true` (fp16 and bf16), `compile=true`, `torchrun --nproc_per_node=1 --launcher pytorch`: each ≥ 8 it without error, ckpt strict-resumable |
| A8 | Export | `convert.py -net aether_small -s 2 --input <A5 ckpt>`: ONNX checker ok; #QuantizeLinear ≥ 1 per FakeQuantize that is enabled; ≥71/74 Conv with input+weight from DequantizeLinear; ORT parity within the S4 QAT tolerance at 40×32 (dynamic shape) |
| A9 | Strip | `--strip-qat` → `test.py` with `[network_g].fused_init=true`, strict load; output vs fake-quant model max ≤ 1e-2 |
| A10 | Cleanup | `grep -rn "aether_core\|from common\|generate_calib_data\|_init_ema\|calibrate(" neosr train.py convert.py test.py` is empty; `options/aether/*.toml` all reach iteration 1 with a temporary dataset path override |

Already demonstrated by scratch probes (not a substitute for A1-A10 on the edited code): §0 table (Opus,
`qat/synth/probe.py`), plus Fable's `qat/{smoke_qat.py,lifecycle.py,g3.py,coverage_fix.py}` (current flow has no FakeQuantize;
`calibrate(iters=)` TypeError; `_init_ema` net lacks `update_parameters`; EMA shape-[1] crash without priming; compile on the prepared net; eager `tq.convert` forward failure).

## 7. Risks, known issues, open questions

- From-scratch QAT trains one fused large-kernel conv instead of lk+sk branches, so the reparam training benefit is lost.
  The recommended recipe is float pretrain → QAT fine-tune. The tomls still run as written.
- Pre-existing and not fixed here: resume loads EMA weights into `net_g` (`image.py:990-995` + `misc.py:157-158`); `n_averaged`
  is dropped (`base.py:348`); the accumulation counter is incremented in both `closure` (`image.py:501-505`) and `optimize_parameters`
  (`:734-738`); schedule-free `optimizer.eval()` runs after the state dict is captured (`base.py:343` vs `:353`);
  `convert.py` fails for every arch today (torch 2.9 defaults to `dynamo=True` and onnxscript is missing). S4 fixes that last one.
- Under DDP, observer and norm stats come from rank 0 (via buffer broadcast). TensorRT engine build is unverified (TRT not installed).
  Multi-GPU DDP and schedule-free optimizers with QAT are unverified.
- x4: the conv after the first PixelShuffle has no input Q/DQ (Fable: 1 of 74). Add a stub after the PixelShuffle if TRT profiling shows it matters.
- Open questions for the owner: (1) is the INT8 target TensorRT, ORT-CPU or both (native CPU INT8 would need the D6 QuantWrapper
  fallback)? (2) Should AMP/compile with QAT be warnings (proposed) or errors? (3) What PSNR/SSIM loss vs the float model is acceptable?
