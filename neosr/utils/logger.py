import datetime
import logging
import time
from typing import Any

from neosr.utils.dist_util import get_dist_info, master_only
from neosr.utils.misc import tc

initialized_logger: dict[Any, Any] = {}


class AvgTimer:
    def __init__(self, window: int = 200) -> None:
        self.window = window  # average window
        self.current_time: float = 0
        self.total_time: float = 0
        self.count: int = 0
        self.avg_time: float = 0
        self.start()

    def start(self) -> None:
        self.start_time = self.tic = time.time()

    def record(self) -> None:
        self.count += 1
        self.toc = time.time()
        self.current_time = self.toc - self.tic
        self.total_time += self.current_time
        # calculate average time
        self.avg_time = self.total_time / self.count

        # reset
        if self.count > self.window:
            self.count = 0
            self.total_time = 0

        self.tic = time.time()

    def get_current_time(self):
        return self.current_time

    def get_avg_time(self):
        return self.avg_time


def get_root_logger(
    logger_name: str = "neosr",
    log_level: int = logging.INFO,
    log_file: str | None = None,
) -> logging.Logger:
    """The "neosr" logger: a stream handler once per process; rank 0 logs at `log_level`, other
    ranks only errors. A `log_file` (rank 0) replaces the previous run's file handler."""
    logger = logging.getLogger(logger_name)
    rank, _ = get_dist_info()
    format_str = "%(asctime)s %(levelname)s: %(message)s"
    if logger_name not in initialized_logger:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter(format_str, datefmt="%d-%m-%Y %I:%M %p |")
        )
        logger.addHandler(stream_handler)
        logger.propagate = False
        initialized_logger[logger_name] = True
    logger.setLevel(log_level if rank == 0 else logging.ERROR)
    if log_file is not None and rank == 0:
        for h in [h for h in logger.handlers if isinstance(h, logging.FileHandler)]:
            logger.removeHandler(h)
            h.close()
        file_handler = logging.FileHandler(log_file, "w", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    return logger


class MessageLogger:
    """Formats the periodic training line. `iter` counts completed updates (T2 §4); `time` is the
    mean seconds per update."""

    def __init__(self, opt: dict[str, Any], tb_logger, start_iter: int = 0) -> None:
        self.exp_name = opt["name"]
        self.start_iter = start_iter
        self.max_iters = opt["logger"].get("total_iter", 1000000)
        self.tb_logger = tb_logger
        self.start_time = time.time()
        self.logger = get_root_logger()

    def reset_start_time(self) -> None:
        self.start_time = time.time()

    @master_only
    def __call__(self, log_vars: dict[str, Any]) -> None:
        epoch: int = log_vars.pop("epoch")
        current_iter: int = int(log_vars.pop("iter"))
        lrs: list[Any] = log_vars.pop("lrs")
        message = (
            f"[ epoch:{epoch:4d} ] {tc.light_green}[ iter:{current_iter:7,d} ]{tc.end}"
        )
        if "time" in log_vars:
            iter_time = log_vars.pop("time")
            done = current_iter - self.start_iter
            time_sec_avg = (time.time() - self.start_time) / done if done > 0 else 0
            eta = datetime.timedelta(
                seconds=int(time_sec_avg * (self.max_iters - current_iter))
            )
            speed = 1 / iter_time if iter_time > 0 else 0.0
            lr = " ".join(f"{v:.2e}" for v in lrs)
            message += (
                f" [ performance: {speed:.3f} it/s ] [ lr: {lr} ] [ eta: {eta} ] "
            )
        for k, v in log_vars.items():
            message += f"[ {k}: {v:.4e} ] "
            if self.tb_logger is not None:
                tag = f"losses/{k}" if k.startswith("l_") else k
                self.tb_logger.add_scalar(tag, v, current_iter)
        self.logger.info(message)


@master_only
def init_tb_logger(log_dir: str):
    from torch.utils.tensorboard import SummaryWriter  # type: ignore # noqa: PLC0415, I001

    return SummaryWriter(log_dir=log_dir)


@master_only
def init_wandb_logger(opt: dict[str, Any]) -> None:
    """We now only use wandb to sync tensorboard log."""
    import wandb  # type: ignore # noqa: PLC0415

    logger = get_root_logger()

    project = opt["logger"]["wandb"]["project"]
    if resume_id := opt["logger"]["wandb"].get("resume_id"):
        wandb_id = resume_id
        resume = "allow"
        logger.warning(f"Resume wandb logger with id={wandb_id}.")
    else:
        wandb_id = wandb.util.generate_id()
        resume = "never"

    wandb.init(
        id=wandb_id,
        resume=resume,
        name=opt["name"],
        config=opt,
        project=project,
        sync_tensorboard=True,
    )

    logger.info(f"Use wandb logger with id={wandb_id}; project={project}.")
