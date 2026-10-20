from typing import Any

import torch
from torch.utils.data import DataLoader


class CUDAPrefetcher:
    """CUDA prefetcher: copies the next batch to the GPU on a side stream.

    Reference: https://github.com/NVIDIA/apex/issues/304#

    It may consume more GPU memory. Call reset() before the first next().

    Args:
    ----
        loader: Dataloader.
        opt (dict): Options.

    """

    def __init__(self, loader: DataLoader, opt: dict[str, Any]) -> None:
        self.ori_loader = loader
        self.opt = opt
        self.stream = torch.cuda.Stream()
        self.device = torch.device("cuda")
        self.batch = None

    def preload(self) -> None:
        try:
            self.batch = next(self.loader)  # self.batch is a dict
        except StopIteration:
            self.batch = None
            return
        # put tensors to gpu
        with torch.cuda.stream(self.stream):  # type: ignore[reportArgumentType]
            for k, v in self.batch.items():
                if torch.is_tensor(v):
                    self.batch[k] = v.to(device=self.device, non_blocking=True)

    def next(self):
        stream = torch.cuda.current_stream()
        stream.wait_stream(self.stream)
        batch = self.batch
        if batch is not None:
            # the tensors were allocated on the side stream: without this, the caching
            # allocator may hand their memory to the next preload while the consumer
            # stream still reads them
            for v in batch.values():
                if torch.is_tensor(v):
                    v.record_stream(stream)
        self.preload()
        return batch

    def reset(self) -> None:
        self.loader = iter(self.ori_loader)
        self.preload()
