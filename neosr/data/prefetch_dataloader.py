from contextlib import nullcontext
from typing import Any

import torch
from torch.utils.data import DataLoader

from neosr.utils.stop import sigint_ignored


class CUDAPrefetcher:
    """Copies the next batch to `device` on a side stream while the current one is consumed.

    Reference: https://github.com/NVIDIA/apex/issues/304#
    The constructor only stores; call reset() at the start of every epoch, then next() until None.
    On a CPU device it is a plain iterator (no stream).
    """

    def __init__(self, loader: DataLoader, device: torch.device) -> None:
        self.loader, self.device = loader, device
        self.stream = torch.cuda.Stream(device) if device.type == "cuda" else None
        self._it = None
        self.batch: dict[str, Any] | None = None

    def reset(self) -> None:
        # workers spawned here inherit SIG_IGN: a terminal Ctrl-C reaches only the trainer (T2 D19)
        with sigint_ignored():
            self._it = iter(self.loader)
        self._preload()

    def _preload(self) -> None:
        try:
            batch = next(self._it)  # type: ignore[arg-type]
        except StopIteration:
            self.batch = None
            return
        ctx = nullcontext() if self.stream is None else torch.cuda.stream(self.stream)
        with ctx:
            self.batch = {
                k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

    def next(self) -> dict[str, Any] | None:
        batch = self.batch
        if self.stream is not None:
            cur = torch.cuda.current_stream(self.device)
            cur.wait_stream(self.stream)
            if batch is not None:
                # the tensors were allocated on the side stream: without this, the caching
                # allocator may hand their memory to the next preload while `cur` still reads them
                for v in batch.values():
                    if torch.is_tensor(v):
                        v.record_stream(cur)
        self._preload()
        return batch
