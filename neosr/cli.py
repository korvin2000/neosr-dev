"""Command-line parsers of the entry points. Parsing only: no config loading, no side effects."""

import argparse
from collections.abc import Sequence


def _parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neosr", description=f"-------- neosr {description} --------"
    )
    parser._optionals.title = "options"
    return parser


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-opt", required=True, help="Path to the option TOML file.")
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm"],
        default="none",
        help="Job launcher.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode: frequent logging and saving."
    )
    # passed by torch.distributed.launch; the device comes from the LOCAL_RANK environment variable
    parser.add_argument(
        "--local_rank", "--local-rank", type=int, default=0, help=argparse.SUPPRESS
    )


def train_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _parser("training")
    _add_run_args(parser)
    parser.add_argument(
        "--auto_resume", action="store_true", help="Resume from the newest state."
    )
    return parser.parse_args(argv)


def infer_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _parser("inference")
    _add_run_args(parser)
    return parser.parse_args(argv)


def convert_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _parser("model conversion")
    add = parser.add_argument
    add("--input", required=True, help="Input PyTorch model path.")
    add("-net", "--network", required=True, help="Generator network.")
    add("-s", "--scale", type=int, default=4, help="Model scale ratio. (default: 4)")
    add(
        "-window",
        "--window",
        type=int,
        default=None,
        help="Window size, for archs that take one.",
    )
    add(
        "--output",
        default=None,
        help="Output ONNX model path (default: the input path with an .onnx suffix); with "
        "--strip-qat the float checkpoint path (default: <input>_float.pth).",
    )
    add(
        "-onnx",
        "--onnx",
        action="store_true",
        help="With --strip-qat: also export the float ONNX. Otherwise ONNX is always exported.",
    )
    add(
        "--strip-qat",
        action="store_true",
        help="From a QAT checkpoint, save a float checkpoint without the QAT modules (load it "
        "with [network_g] fused_init = true).",
    )
    add(
        "-prune",
        "--prune",
        action="store_true",
        help="Enables network prune if supported.",
    )
    add("-opset", "--opset", type=int, default=17, help="ONNX opset. (default: 17)")
    add(
        "-static",
        "--static",
        type=int,
        nargs=3,
        default=None,
        metavar=("C", "H", "W"),
        help="Static shape for ONNX conversion. Example: -static 3 640 640",
    )
    add(
        "-nocheck",
        "--nocheck",
        action="store_true",
        help="Disables checking against the original PyTorch model on ONNX conversion.",
    )
    add(
        "-fp16",
        "--fp16",
        action="store_true",
        help="Enable half-precision. (default: false)",
    )
    add("-optimize", "--optimize", action="store_true", help="Run ONNX optimizations.")
    add(
        "-fulloptimization",
        "--fulloptimization",
        action="store_true",
        help="Run full ONNX optimizations.",
    )
    return parser.parse_args(argv)
