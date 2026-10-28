"""Export a generator to ONNX (design T3 D26): CPU end to end, no Runtime; the scale comes from -s.
Exit codes: 0 ok, 1 unexpected (traceback logged), 2 ConfigError (e.g. unknown network)."""

import argparse
import logging
import sys
import warnings
from collections.abc import Sequence
from os import path as osp
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.ao.quantization import FakeQuantizeBase

from neosr.archs import build_network
from neosr.archs.arch_util import DySample
from neosr.cli import convert_args
from neosr.config import ConfigError
from neosr.models.checkpoint import load_weights, read_weights
from neosr.utils import get_root_logger

log = logging.getLogger("neosr")

"""
TODO

def rep_rt4ksr(net):
    rep_model = net.eval()
    rep_state_dict = net.eval().state_dict()
    pretrained_state_dict = net.train().state_dict()

    for k, v in rep_state_dict.items():
        if "rep_conv.weight" in k:
            # merge conv1x1-conv3x3-conv1x1
            k0 = pretrained_state_dict[k.replace("rep", "expand")]
            k1 = pretrained_state_dict[k.replace("rep", "fea")]
            k2 = pretrained_state_dict[k.replace("rep", "reduce")]

            bias_str = k.replace("weight", "bias")
            b0 = pretrained_state_dict[bias_str.replace("rep", "expand")]
            b1 = pretrained_state_dict[bias_str.replace("rep", "fea")]
            b2 = pretrained_state_dict[bias_str.replace("rep", "reduce")]

            mid_feats, n_feats = k0.shape[:2]

            # first step: remove the middle identity
            for i in range(mid_feats):
                k1[i, i, 1, 1] += 1.0

            # second step: merge the first 1x1 convolution and the next 3x3 convolution
            merged_k0k1 = F.conv2d(input=k1, weight=k0.permute(1, 0, 2, 3))
            merged_b0b1 = b0.view(1, -1, 1, 1) * torch.ones(1, mid_feats, 3, 3, device=b0.device)
            merged_b0b1 = F.conv2d(input=merged_b0b1, weight=k1, bias=b1)

            # third step: merge the remain 1x1 convolution
            merged_k0k1k2 = F.conv2d(input=merged_k0k1.permute(1, 0, 2, 3), weight=k2).permute(1, 0, 2, 3)
            merged_b0b1b2 = F.conv2d(input=merged_b0b1, weight=k2, bias=b2).view(-1)

            # last step: remove the global identity
            for i in range(n_feats):
                merged_k0k1k2[i, i, 1, 1] += 1.0

            # save merged weights and biases in rep state dict
            rep_state_dict[k] = merged_k0k1k2.float()
            rep_state_dict[bias_str] = merged_b0b1b2.float()

        elif "rep_conv.bias" in k:
            pass

        elif k in pretrained_state_dict.keys():
            rep_state_dict[k] = pretrained_state_dict[k]

    rep_model.load_state_dict(rep_state_dict, strict=True)
    return rep_model
"""


def build_net(args: argparse.Namespace, **kw) -> nn.Module:
    net_opt = {"type": args.network, **kw}
    if args.window:
        net_opt["window_size"] = args.window
    return build_network(net_opt, scale=args.scale)


def load_net(args: argparse.Namespace, *, qat_only: bool = False) -> nn.Module:
    """Build `args.network` at `args.scale` on the CPU and load `args.input` (any read_weights
    layout) into the topology it was saved in: QAT, deploy (fused_init) or trained."""
    print(f"\n-------- Attempting to build network [{args.network}].")
    net = build_net(args)
    sd = read_weights(args.input)
    qat_state = getattr(net, "qat_state", None)
    kw = qat_state(sd) if qat_state is not None else None
    if qat_only and kw is None:
        msg = f"--strip-qat needs a QAT checkpoint; {args.input} has no fake-quant modules"
        raise ConfigError(msg)
    if kw is not None:
        net.prepare_qat(**kw)  # the QAT topology exists before its tensors load (T5 §2)
    elif hasattr(net, "fuse_model") and sd.keys() != net.state_dict().keys():
        fused = build_net(
            args, fused_init=True
        )  # a deploy checkpoint, e.g. from --strip-qat
        if sd.keys() == fused.state_dict().keys():
            net = fused
    load_weights(net, sd, True, get_root_logger())
    net.eval()
    if is_qat(net):  # QDQ export: frozen ranges, fake quant on, folded norms
        net.set_qat_phase(fake_quant=True, observe=False, freeze_norm=True)
        net.fold_norms()
    if args.prune:  # plainusr's block deploy is broken: export the trained form
        log.warning("--prune: no network supports reparametrization; exported unfused")
    print(f"-------- Successfully loaded network [{args.network}].")
    return net


def is_qat(net: nn.Module) -> bool:
    return any(isinstance(m, FakeQuantizeBase) for m in net.modules())


def strip_qat(args: argparse.Namespace) -> nn.Module:
    """--strip-qat (T5 D20, A9): the float deploy net of a QAT checkpoint (reparam convs fused, norms
    folded, every fake-quant module dropped), saved as {"params": ...} to `args.output`. It loads
    strictly into the network built with [network_g] fused_init = true."""
    qat = load_net(args, qat_only=True)
    fq = tuple(
        f"{n}." for n, m in qat.named_modules() if isinstance(m, FakeQuantizeBase)
    )
    net = build_net(args, fused_init=True).eval()
    net.load_state_dict({
        k: v for k, v in qat.state_dict().items() if not k.startswith(fq)
    })
    # fake quant off, the QAT net is the float net (same output); A9: vs fake quant max <= 1e-2
    if not args.nocheck:
        x = torch.rand(1, 3, 40, 32, generator=torch.Generator().manual_seed(0))
        with torch.inference_mode():
            quant = qat(x)
            qat.set_qat_phase(fake_quant=False, observe=False, freeze_norm=True)
            ref, out = qat(x), net(x)
        err = (out - ref).abs().max().item()
        gap = (out - quant).abs()
        print(
            f"-------- float vs fake-quant: max {gap.max():.2e}, mean {gap.mean():.2e}"
        )
        if not err <= 1e-4:  # NaN fails too
            msg = f"stripped net deviates from the QAT net without fake quant: max {err:.2e}"
            raise RuntimeError(msg)
        if not gap.max().item() <= 1e-2:
            msg = f"float net deviates from the fake-quant net: max {gap.max():.2e} > 1e-2"
            raise RuntimeError(msg)
    torch.save({"params": net.state_dict()}, args.output)
    print(f"-------- Float checkpoint saved at: {args.output}")
    return net


def scalarize_qdq(path: str) -> None:
    """ORT fuses QDQ patterns (QLinearAdd/Mul) only with 0-d per-tensor scale and zero point."""
    import onnx  # noqa: PLC0415

    mdl = onnx.load(path)
    inits = {i.name: i for i in mdl.graph.initializer}
    alias = {n.output[0]: n.input[0] for n in mdl.graph.node if n.op_type == "Identity"}
    for node in mdl.graph.node:
        if node.op_type in {"QuantizeLinear", "DequantizeLinear"}:
            for name in node.input[1:3]:
                root = name  # constant folding shares initializers through Identity
                while root in alias:
                    root = alias[root]
                if root in inits and list(inits[root].dims) == [1]:
                    del inits[root].dims[:]
    onnx.save(mdl, path)


TRACE = (1, 3, 20, 20)  # dynamic exports are traced at this input


def export_shapes(args: argparse.Namespace) -> list[tuple[int, ...]]:
    """Static: the trace shape. Dynamic: also a rectangular size of its own and batch 2 (MT-02)."""
    if args.static is not None:
        return [(1, *args.static)]
    _, c, h, w = TRACE
    return [TRACE, (1, c, h + 12, w + 4), (2, c, h, w)]


def _outputs(args: argparse.Namespace, sessions, torch_model, sample, *, fp16=False):
    """(shape label, torch output, ORT output per session) at every export_shapes() input. A shape
    the torch model itself rejects is skipped with a warning (the arch does not support it); an ORT
    error or an output shape that differs from torch raises naming the shape: a dynamic graph frozen
    at the trace shape passes there and is silently wrong elsewhere (MT-02)."""
    hint = "use -static C H W" if args.static is None else "the arch does not export"
    feed = sessions[0].get_inputs()[0].name
    gen = torch.Generator().manual_seed(0)
    for i, shape in enumerate(export_shapes(args)):
        x = sample(shape, generator=gen)
        at = "x".join(map(str, shape))
        try:
            with torch.inference_mode():
                ref = torch_model(x).detach().cpu().numpy()
        except Exception as e:
            if i == 0:  # the trace shape ran during export
                raise
            log.warning(f"verification skips {at}: the torch model fails there ({e})")
            continue
        xin = x.numpy().astype(np.float16 if fp16 else np.float32)
        outs = []
        for sess in sessions:
            try:
                out = sess.run(None, {feed: xin})[0].astype(np.float32)
            except Exception as e:
                msg = f"ONNX export fails at {at} in onnxruntime ({e}): {hint}"
                raise RuntimeError(msg) from e
            if out.shape != ref.shape:
                msg = f"ONNX export is wrong at {at}: output {out.shape}, torch {ref.shape}; {hint}"
                raise RuntimeError(msg)
            outs.append(out)
        yield at, ref, outs


def assert_verify(
    args: argparse.Namespace, onnx_model, torch_model, *, fp16: bool = False
) -> None:
    """torch vs onnxruntime (CPU) at every export_shapes() input (_outputs). Float: rtol 1e-2, atol
    1e-3 (looser than numpy's 1e-7/0: ORT reorders reductions); fp16 (its fp16 input vs the fp32
    torch output): max 2e-2, mean 2e-3 abs."""
    import onnxruntime  # noqa: PLC0415

    # NOTE: "CUDAExecutionProvider" errors if some nvidia libs are not found
    sess = onnxruntime.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])
    hint = "use -static C H W" if args.static is None else "the arch does not export"
    for at, ref, (out,) in _outputs(args, [sess], torch_model, torch.randn, fp16=fp16):
        # first: a NaN/inf on either side is its own failure, not a tolerance question
        if not (np.isfinite(ref).all() and np.isfinite(out).all()):
            msg = f"ONNX verification failed at {at}: non-finite output (torch or onnxruntime)"
            raise RuntimeError(msg)
        err = np.abs(out - ref)
        ok = (
            err.max() <= 2e-2 and err.mean() <= 2e-3
            if fp16
            else (err <= 1e-3 + 1e-2 * np.abs(ref)).all()
        )
        if not ok:
            msg = (
                f"ONNX export is wrong at {at} (max err {err.max():.2e}, mean {err.mean():.2e}): "
                f"this arch/shape is not export-safe; {hint}"
            )
            raise RuntimeError(msg)
    print(
        f"-------- Model successfully verified at {len(export_shapes(args))} shape(s)."
    )


def verify_qat(args: argparse.Namespace, onnx_model, torch_model) -> None:
    """Fake-quant torch vs ORT at every export_shapes() input (_outputs), on [0, 1) inputs. Rounding
    ties flip single int8 steps, so QAT has its own tolerance (T5 S4): unoptimized max <= 5e-3, mean
    <= 1e-4; ORT's default (fused int8 kernels) mean <= 2e-3."""
    import onnxruntime  # noqa: PLC0415

    level = onnxruntime.GraphOptimizationLevel
    levels = (
        (level.ORT_DISABLE_ALL, 5e-3, 1e-4),
        (level.ORT_ENABLE_ALL, float("inf"), 2e-3),
    )
    sessions = []
    for opt_level, *_ in levels:
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = opt_level
        sessions.append(
            onnxruntime.InferenceSession(
                onnx_model, so, providers=["CPUExecutionProvider"]
            )
        )
    for at, ref, outs in _outputs(args, sessions, torch_model, torch.rand):
        for (opt_level, max_tol, mean_tol), out in zip(levels, outs, strict=True):
            err = np.abs(out - ref)
            print(
                f"-------- QDQ parity at {at} ({opt_level.name}): "
                f"max {err.max():.2e}, mean {err.mean():.2e}"
            )
            # NaN compares False: a non-finite output (either side) fails outright
            if (
                not np.isfinite(err).all()
                or err.max() > max_tol
                or err.mean() > mean_tol
            ):
                msg = f"QDQ ONNX deviates from the fake-quant model at {at} ({opt_level.name})"
                raise RuntimeError(msg)
    print(
        f"-------- Model successfully verified at {len(export_shapes(args))} shape(s)."
    )


# Known export gaps (MT-17; re-swept after the stage-2 arch lanes, opset 17, TorchScript exporter),
# refused before anything is built or written. An arch that becomes exportable leaves the table.
NO_ONNX = {
    "atd": "uses the Sort operator",
    "atd_light": "uses the Sort operator",
    "flexnet": "aten::rms_norm has no opset-17 export",
    "metaflexnet": "aten::rms_norm has no opset-17 export",
    "esc": "its padding cannot be traced",
    "esc_light": "its padding cannot be traced",
    "esc_fp": "its padding cannot be traced",
    "esc_large": "its padding cannot be traced",
    "catanet": "Unfold needs a known input size, also with -static",
    "lmlt": "adaptive_max_pool2d output sizes are not constant, also with -static",
    "lmlt_tiny": "adaptive_max_pool2d output sizes are not constant, also with -static",
    "lmlt_large": "adaptive_max_pool2d output sizes are not constant, also with -static",
    "asid": "its window padding exports with the wrong size, also with -static",
    "asid_d8": "its window padding exports with the wrong size, also with -static",
    "omnisr": "its window padding exports with the wrong size, also with -static",
}
STATIC_ONLY = {  # the dynamic graph is frozen at the trace shape
    "safmn": "adaptive_max_pool2d output sizes depend on the input",
    "safmn_l": "adaptive_max_pool2d output sizes depend on the input",
    "light_safmnpp": "adaptive_max_pool2d output sizes depend on the input",
    "msdan": "adaptive_max_pool2d output sizes depend on the input",
    "ditn": "its patch reshape freezes the input size",
    "dctlsa": "its window merge freezes the batch size",
    "grformer": "its window merge freezes the batch size",
    "grformer_medium": "its window merge freezes the batch size",
    "grformer_large": "its window merge freezes the batch size",
    "microsr": "its window merge freezes the batch size",
    "microsr_light": "its window merge freezes the batch size",
    "mfghmoe": "its window merge freezes the batch size",
    "srformer_medium": "a 20x20 trace is one 24 px window: the shift mask freezes",
}


def check_export(args: argparse.Namespace) -> None:
    """Every refusal the converter can know up front (MT-17), before a file is written."""
    if args.fulloptimization:
        msg = "--fulloptimization (onnxsim) is not available; -optimize runs the ORT optimizer"
        raise ConfigError(msg)
    if args.network in NO_ONNX:
        msg = f"[{args.network}] cannot be exported to ONNX: {NO_ONNX[args.network]}"
        raise ConfigError(msg)
    if args.static is None and args.network in STATIC_ONLY:
        msg = (
            f"[{args.network}] exports only with a static shape "
            f"({STATIC_ONLY[args.network]}): add -static C H W"
        )
        raise ConfigError(msg)
    try:
        import onnx  # noqa: F401, PLC0415
        import onnxruntime  # noqa: F401, PLC0415

        if args.fp16:  # ships with onnxruntime
            import onnxruntime.transformers.float16  # noqa: F401, PLC0415
    except ImportError as e:
        msg = f"ONNX export needs onnx and onnxruntime (with its float16 converter for -fp16): {e}"
        raise ConfigError(msg) from e


def _module_path(node_name: str) -> str:
    """The qualified module name of a TorchScript-exported node ("/a/a.0/up/Conv" -> "a.0.up"): a
    scope segment is the full name below a container, else the name relative to its parent."""
    path = ""
    for seg in node_name.split("/")[1:-1]:
        path = seg if not path or seg.startswith(path + ".") else f"{path}.{seg}"
    return path


def _cast_to(node) -> int:
    return next(a.i for a in node.attribute if a.name == "to")


def to_fp16(src: str, dst: str, fp32_modules: Sequence[str] = ()) -> None:
    """A float16 graph: weights, compute and the input/output tensors (MT-16), with ORT's own
    converter (onnxconverter-common is unmaintained); ops without fp16 kernels get casts. The nodes
    of `fp32_modules` (qualified names) and every GridSample stay fp32: DySample's sampling grid
    holds pixel coordinates that fp16 rounds to whole pixels at widths of 1-2k."""
    import onnx  # noqa: PLC0415
    from onnxruntime.transformers.float16 import (  # noqa: PLC0415
        DEFAULT_OP_BLOCK_LIST,
        convert_float_to_float16,
    )
    from onnxruntime.transformers.onnx_model import OnnxModel  # noqa: PLC0415

    graph = onnx.load(src)
    keep = [
        n.name
        for n in graph.graph.node
        if any(
            not q or (p := _module_path(n.name)) == q or p.startswith(q + ".")
            for q in fp32_modules
        )
    ]
    model = OnnxModel(
        convert_float_to_float16(
            graph,
            op_block_list=[*DEFAULT_OP_BLOCK_LIST, "GridSample"],
            node_block_list=keep,
        )
    )
    # the converter casts every kept node's output to fp16 and back for the next kept node: skip the
    # fp16 round trip (only Cast(fp16) -> Cast(fp32) pairs, so no int or other casts collapse)
    f16, f32 = onnx.TensorProto.FLOAT16, onnx.TensorProto.FLOAT
    producer = model.output_name_to_node()
    for n in model.nodes():
        p = producer.get(n.input[0]) if n.op_type == "Cast" else None
        if (
            p is not None
            and p.op_type == "Cast"
            and (_cast_to(p), _cast_to(n)) == (f16, f32)
        ):
            n.input[0] = p.input[0]
    model.prune_graph()
    model.topological_sort()
    onnx.save(model.model, dst)
    onnx.checker.check_model(onnx.load(dst))


def to_onnx(args: argparse.Namespace, model: nn.Module | None = None) -> None:
    check_export(args)  # first: a missing onnx/onnxruntime is a ConfigError (exit 2)
    import onnx  # noqa: PLC0415
    import onnxruntime  # noqa: PLC0415

    if model is None:
        model = load_net(args)
    # set model to eval mode
    model.eval()
    qat = is_qat(model)
    if qat and args.opset < 13:
        msg = f"a QAT model needs --opset >= 13 (per-channel QuantizeLinear), got {args.opset}"
        raise ConfigError(msg)
    if qat and args.fp16:
        msg = "-fp16 does not apply to a QAT (int8 QDQ) export"
        raise ConfigError(msg)

    # set static or dynamic
    dummy_input = torch.randn(export_shapes(args)[0])
    names = {"input_names": ["input"], "output_names": ["output"]}
    if args.static is None:
        # distinct dim names: a shared name tells the runtime the dims are equal (MT-07)
        names["dynamic_axes"] = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "out_height", 3: "out_width"},
        }

    # add _fp32 suffix to output str
    filename, extension = osp.splitext(args.output)  # noqa: PTH122
    output_fp32 = filename + "_fp32" + extension
    # begin conversion
    print("-------- Starting ONNX conversion (this can take a while)...")

    with torch.inference_mode(), torch.device("cpu"):
        torch.onnx.export(
            model,
            dummy_input,
            output_fp32,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=qat,  # QAT: Q/DQ scale/zero point become initializers
            dynamo=False,  # the TorchScript exporter: torch>=2.9 defaults to dynamo (needs onnxscript)
            **names,  # type: ignore[arg-type]
        )

    print("-------- Conversion was successful. Verifying...")
    if qat:
        scalarize_qdq(output_fp32)
    # verify onnx
    onnx.checker.check_model(onnx.load(output_fp32))
    print(
        f"-------- Model successfully converted to ONNX format. Saved at: {output_fp32}."
    )
    # verify outputs
    if args.nocheck is False:
        (verify_qat if qat else assert_verify)(args, output_fp32, model)

    if args.optimize:
        print("-------- Running ONNX optimization...")
        output_optimized = filename + "_fp32_optimized" + extension
        session_opt = onnxruntime.SessionOptions()
        # ENABLE_ALL can cause compatibility issues, leaving EXTENDED as default
        session_opt.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
        session_opt.optimized_model_filepath = output_optimized
        # save
        onnxruntime.InferenceSession(output_fp32, session_opt)
        # verify
        onnx.checker.check_model(onnx.load(output_optimized))
        print(f"-------- Model successfully optimized. Saved at: {output_optimized}")

    if args.fp16:  # from the portable fp32 graph, not the ORT-optimized one
        print("-------- Converting to fp16...")
        output_fp16 = filename + "_fp16" + extension
        dysample = [n for n, m in model.named_modules() if isinstance(m, DySample)]
        to_fp16(output_fp32, output_fp16, dysample)
        if args.nocheck is False:
            assert_verify(args, output_fp16, model, fp16=True)
        print(f"-------- Half-precision model saved at: {output_fp16}.")


def main(argv: Sequence[str] | None = None) -> int:
    args = convert_args(argv)  # argparse may SystemExit(2)
    if args.output is None:
        inp = Path(args.input)
        out = (
            inp.with_name(f"{inp.stem}_float.pth")
            if args.strip_qat
            else inp.with_suffix(".onnx")
        )
        args.output = str(out)
    get_root_logger()
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        if not args.strip_qat:
            to_onnx(args)
        else:
            if args.onnx:  # refuse before the float checkpoint is written
                check_export(args)
            net = strip_qat(args)
            if args.onnx:  # the float ONNX of the stripped net, next to its checkpoint
                args.output = str(Path(args.output).with_suffix(".onnx"))
                to_onnx(args, net)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    except Exception:
        log.exception("conversion failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
