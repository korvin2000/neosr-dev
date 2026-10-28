import signal
from collections.abc import Generator
from contextlib import contextmanager

import torch
import torch.distributed as dist


class StopFlag:
    """1st SIGINT: request a stop at an update boundary. 2nd: KeyboardInterrupt (exit 130).

    Under DDP every boundary reads the MAX-reduction started at the previous boundary and starts the
    next one, so all ranks stop at the same update (one after the signal's boundary) without a
    collective wait or a host sync per update (T-11)."""

    def __init__(self) -> None:
        self._set = False
        self._pending: tuple | None = None  # DDP: (work, host value, cuda event | None)

    def install(self) -> None:
        signal.signal(signal.SIGINT, self._on_sigint)

    def _on_sigint(self, *_) -> None:  # no logging/IO in the handler
        self._set = True
        signal.signal(signal.SIGINT, signal.default_int_handler)

    def requested(self, device: torch.device) -> bool:
        if not dist.is_initialized():
            return self._set
        out = False
        if self._pending is not None:
            work, host, done = self._pending
            if done is None:
                work.wait()  # CPU tensors: the collective ran during the update
            else:
                done.synchronize()  # its copy was queued an update ago: no stall
            out = bool(host)
        t = torch.full((1,), float(self._set), device=device)  # a fill, no host copy
        work = dist.all_reduce(t, op=dist.ReduceOp.MAX, async_op=True)
        if device.type == "cuda":
            work.wait()  # NCCL: the current stream waits for it, the host does not
            host = torch.empty(1, pin_memory=True).copy_(t, non_blocking=True)
            done = torch.cuda.Event()
            done.record()
            self._pending = (work, host, done)
        else:
            self._pending = (work, t, None)
        return out


@contextmanager
def sigint_ignored() -> Generator[None]:
    """Processes spawned inside inherit SIG_IGN across exec, so a terminal Ctrl-C reaches only the trainer."""
    prev = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, prev)
