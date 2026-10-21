import signal
from collections.abc import Generator
from contextlib import contextmanager

import torch
import torch.distributed as dist


class StopFlag:
    """1st SIGINT: request a stop at the next update boundary. 2nd: KeyboardInterrupt (exit 130)."""

    def __init__(self) -> None:
        self._set = False

    def install(self) -> None:
        signal.signal(signal.SIGINT, self._on_sigint)

    def _on_sigint(self, *_) -> None:  # no logging/IO in the handler
        self._set = True
        signal.signal(signal.SIGINT, signal.default_int_handler)

    def requested(self, device: torch.device) -> bool:
        if not dist.is_initialized():
            return self._set
        t = torch.tensor(float(self._set), device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)  # all ranks leave together
        return bool(t.item())


@contextmanager
def sigint_ignored() -> Generator[None]:
    """Processes spawned inside inherit SIG_IGN across exec, so a terminal Ctrl-C reaches only the trainer."""
    prev = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, prev)
