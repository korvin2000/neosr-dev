"""Inference entry point (design T3 §3.7): run the generator over every dataset of an inference config.
Exit codes as train.py: 0 ok, 1 unexpected (traceback logged), 2 ConfigError, 130 Ctrl-C."""

import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from neosr.cli import infer_args
from neosr.config import ConfigError, load_config
from neosr.data import build_dataloader, build_dataset
from neosr.models import build_model
from neosr.runtime import Runtime, setup_runtime
from neosr.utils import get_root_logger, get_time_str, make_exp_dirs, tc

ROOT = Path(__file__).resolve().parent
log = logging.getLogger("neosr")


def test(cfg: dict[str, Any], rt: Runtime) -> Any:
    """Build the loaders and the eager inference model, then validate on each dataset. Returns the model."""
    make_exp_dirs(cfg)
    p = cfg["path"]
    get_root_logger(
        log_file=str(Path(p["log"]) / f"test_{cfg['name']}_{get_time_str()}.log")
    )
    loaders = []
    for phase, ds_cfg in sorted(cfg["datasets"].items()):
        loader = build_dataloader(build_dataset(ds_cfg), ds_cfg, runtime=rt)
        if len(loader) == 0:
            msg = f"[datasets.{phase}] has no images"
            raise ConfigError(msg)
        log.info(f"Number of test images in {ds_cfg['name']}: {len(loader.dataset)}")  # type: ignore[arg-type]
        loaders.append(loader)
    model = build_model(cfg, rt)
    save_img = (cfg.get("val") or {}).get("save_img", True)
    for loader in loaders:
        log.info(f"Testing {loader.dataset.opt['name']}...")  # type: ignore[attr-defined]
        start = time.perf_counter()
        model.validation(
            loader, current_iter=cfg["name"], tb_logger=None, save_img=save_img
        )
        took = time.perf_counter() - start
        fps = len(loader.dataset) / took  # type: ignore[arg-type]
        log.info(
            f"{tc.light_green}Inference took {took:.2f} seconds, at {fps:.2f} fps.{tc.end}"
        )
    return model


def main(argv: Sequence[str] | None = None) -> int:
    args = infer_args(argv)  # argparse may SystemExit(2)
    get_root_logger()
    try:
        cfg = load_config(args.opt, is_train=False, root=ROOT, debug=args.debug)
        rt = setup_runtime(cfg, launcher=args.launcher)
        test(cfg, rt)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        log.error("interrupted")
        return 130
    except Exception:
        log.exception("inference failed")
        return 1
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
