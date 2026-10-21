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


def load_net(args: argparse.Namespace) -> nn.Module:
    """Build `args.network` at `args.scale` on the CPU and load `args.input` (any read_weights layout)."""
    print(f"\n-------- Attempting to build network [{args.network}].")
    net_opt = {"type": args.network}
    if args.window:
        net_opt["window_size"] = args.window
    net = build_network(net_opt, scale=args.scale)
    sd = read_weights(args.input)
    qat_state = getattr(net, "qat_state", None)
    if qat_state is not None and (kw := qat_state(sd)) is not None:
        net.prepare_qat(**kw)  # the QAT topology exists before its tensors load (T5 §2)
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


def verify_qat(args: argparse.Namespace, onnx_model, torch_model) -> None:
    """Fake-quant torch vs ORT. Rounding ties flip single int8 steps, so QAT has its own tolerance
    (T5 S4): unoptimized max <= 5e-3, mean <= 1e-4; ORT's default (fused int8 kernels) mean <= 2e-3."""
    import onnxruntime  # noqa: PLC0415

    shape = (1, *args.static) if args.static is not None else (1, 3, 40, 32)
    x = torch.rand(shape, generator=torch.Generator().manual_seed(0))
    with torch.inference_mode():
        ref = torch_model(x).numpy()
    level = onnxruntime.GraphOptimizationLevel
    for opt_level, max_tol, mean_tol in (
        (level.ORT_DISABLE_ALL, 5e-3, 1e-4),
        (level.ORT_ENABLE_ALL, float("inf"), 2e-3),
    ):
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = opt_level
        sess = onnxruntime.InferenceSession(
            onnx_model, so, providers=["CPUExecutionProvider"]
        )
        out = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
        err = np.abs(out - ref)
        print(
            f"-------- QDQ parity ({opt_level.name}): max {err.max():.2e}, mean {err.mean():.2e}"
        )
        # NaN compares False: a non-finite output (either side) fails outright
        if not np.isfinite(err).all() or err.max() > max_tol or err.mean() > mean_tol:
            msg = f"QDQ ONNX deviates from the fake-quant model ({opt_level.name})"
            raise RuntimeError(msg)
    print("-------- Model successfully verified.")


def assert_verify(args: argparse.Namespace, onnx_model, torch_model) -> None:
    import onnxruntime  # noqa: PLC0415

    if args.static is not None:
        dummy_input = torch.randn(1, *args.static, requires_grad=True)
    else:
        dummy_input = torch.randn(1, 3, 20, 20, requires_grad=True)
    # onnxruntime output prediction
    # NOTE: "CUDAExecutionProvider" errors if some nvidia libs
    # are not found, defaulting to cpu
    ort_session = onnxruntime.InferenceSession(
        onnx_model, providers=["CPUExecutionProvider"]
    )
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.detach().cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    # torch outputs
    with torch.inference_mode():
        torch_outputs = torch_model(dummy_input)

    # final assert - default tolerance values - rtol=1e-03, atol=1e-05
    np.testing.assert_allclose(
        torch_outputs.detach().cpu().numpy(), ort_outs[0], rtol=0.01, atol=0.001
    )
    print("-------- Model successfully verified.")


def to_onnx(args: argparse.Namespace) -> None:
    import onnx  # noqa: PLC0415
    import onnxruntime  # noqa: PLC0415

    # error if network can't be converted
    net_error = ["craft", "ditn"]
    if args.network in net_error:
        msg = f"Network [{args.network}] cannot be converted to ONNX."
        raise RuntimeError(msg)

    # load network and send to device
    model = load_net(args)
    # set model to eval mode
    model.eval()
    qat = is_qat(model)
    if qat and args.opset < 13:
        msg = f"a QAT model needs --opset >= 13 (per-channel QuantizeLinear), got {args.opset}"
        raise ConfigError(msg)

    # set static or dynamic
    if args.static is not None:
        dummy_input = torch.randn(1, *args.static, requires_grad=True)
    else:
        dummy_input = torch.randn(1, 3, 20, 20, requires_grad=True)

    # dict for dynamic axes
    if args.static is None:
        dyn_axes = {
            "dynamic_axes": {
                "input": {0: "batch_size", 2: "width", 3: "height"},
                "output": {0: "batch_size", 2: "width", 3: "height"},
            },
            "input_names": ["input"],
            "output_names": ["output"],
        }
    else:
        dyn_axes = {"input_names": ["input"], "output_names": ["output"]}

    # add _fp32 suffix to output str
    filename, extension = osp.splitext(args.output)  # noqa: PTH122
    output_fp32 = filename + "_fp32" + extension
    # begin conversion
    print("-------- Starting ONNX conversion (this can take a while)...")

    with torch.inference_mode(), torch.device("cpu"):
        # TODO: add param dynamo=True as a switch
        # py2.5 supports the verify=True flag now as well
        torch.onnx.export(
            model,
            dummy_input,
            output_fp32,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=qat,  # QAT: Q/DQ scale/zero point become initializers
            dynamo=False,  # the TorchScript exporter: torch>=2.9 defaults to dynamo (needs onnxscript)
            **(dyn_axes or {}),  # type: ignore
        )

    print("-------- Conversion was successful. Verifying...")
    if qat:
        scalarize_qdq(output_fp32)
    # verify onnx
    load_onnx = onnx.load(output_fp32)
    onnx.checker.check_model(load_onnx)
    print(
        f"-------- Model successfully converted to ONNX format. Saved at: {output_fp32}."
    )
    # verify outputs
    if args.nocheck is False:
        (verify_qat if qat else assert_verify)(args, output_fp32, model)

    if args.optimize:
        print("-------- Running ONNX optimization...")
        filename, extension = osp.splitext(args.output)  # noqa: PTH122
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

    if args.fp16:
        print(
            "-------- fp16 conversion has been temporarily disabled due to onnxconverter-common not being maintained."
        )
        """
        print("-------- Converting to fp16...")
        output_fp16 = filename + "_fp16" + extension
        # convert to fp16
        if args.optimize:
            to_fp16 = convert_float_to_float16(onnx.load(output_optimized))  # type: ignore[reportPossiblyUnboundVariable]
        else:
            to_fp16 = convert_float_to_float16(load_onnx)
        # save
        onnx.save(to_fp16, output_fp16)
        # verify
        onnx.checker.check_model(onnx.load(output_fp16))
        print(
            f"-------- Model successfully converted to half-precision. Saved at: {output_fp16}."
        )
        """

    if args.fulloptimization:
        msg = "ONNXSimplify has been temporarily disabled."
        raise ValueError(msg)
        """
        # error if network can't run through onnxsim
        opt_error = ["omnisr"]
        if args.network in opt_error:
            msg = f"Network [{args.network}] doesnt support full optimization."
            raise RuntimeError(msg)

        print("-------- Running full optimization (this can take a while)...")
        output_fp32_fulloptimized = filename + "_fp32_fullyoptimized" + extension
        output_fp16_fulloptimized = filename + "_fp16_fullyoptimized" + extension

        # run onnxsim
        if args.optimize:
            simplified, check = simplify(onnx.load(output_optimized))
        elif args.fp16:
            simplified, check = simplify(onnx.load(output_fp16))
        else:
            simplified, check = simplify(load_onnx)
        assert check, "Couldn't validate ONNX model."

        # save and verify
        if args.fp16:
            onnx.save(simplified, output_fp16_fulloptimized)
            onnx.checker.check_model(onnx.load(output_fp16_fulloptimized))
        else:
            onnx.save(simplified, output_fp32_fulloptimized)
            onnx.checker.check_model(onnx.load(output_fp32_fulloptimized))

        print(
            f"-------- Model successfully optimized. Saved at: {output_fp32_fulloptimized}\n"
        )
        """


def main(argv: Sequence[str] | None = None) -> int:
    args = convert_args(argv)  # argparse may SystemExit(2)
    if args.output is None:
        args.output = str(Path(args.input).with_suffix(".onnx"))
    get_root_logger()
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        to_onnx(args)
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
