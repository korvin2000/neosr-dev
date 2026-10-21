"""On-the-fly degradation trainer: `image` whose training lq is synthesised from gt (design T2 §8.5)."""

from typing import Any

import torch

from neosr.data.otf_degradation import Degrader, PairPool
from neosr.data.transforms import paired_random_crop
from neosr.models.image import image
from neosr.runtime import Runtime


class otf(image):
    """Real-ESRGAN two-stage degradation plus the training pair pool; validation is paired, as in image."""

    def __init__(self, opt: dict[str, Any], runtime: Runtime) -> None:
        super().__init__(opt, runtime)
        if self.is_train:  # inference never needs the degradation keys
            # [degradations] is already merged into datasets.train by load_config
            cfg = opt["datasets"]["train"]
            self.patch_size = cfg["patch_size"]
            self.degrader = Degrader(cfg, scale=self.scale, device=self.device)
            self.pool = PairPool(
                cfg.get("queue_size", 180), cfg["batch_size"], device=self.device
            )

    @torch.no_grad()  # not inference_mode: the batch must stay usable by autograd
    def feed_data(self, data: dict[str, Any]) -> None:
        gt = data["gt"].to(self.device, non_blocking=True)
        lq = self.degrader(gt, data["kernel1"], data["kernel2"], data["sinc_kernel"])
        # patch_size is the lq patch size
        gt, lq = paired_random_crop([gt], lq, self.patch_size, self.scale)
        self._set_batch(*self.pool(lq, gt))
