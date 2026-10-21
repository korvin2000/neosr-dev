import numpy as np
import pywt
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class AdvView(nn.Module):
    """The adversarial view: identity before start_iter, then the 3 scaled SWT high-frequency bands of Y.
    fp32 and differentiable, built once; the same op for real and fake, so D gets no precision cue.
    """

    def __init__(self, start_iter: int = 0) -> None:
        super().__init__()
        self.start_iter = start_iter
        w = pywt.Wavelet("sym19")  # type: ignore[reportAttributeAccessIssue]
        # analysis bank (unit-DC lo, hi), flipped since conv2d correlates; constants, not state
        lo = np.divide(w.dec_lo, sum(w.dec_lo))
        bank = torch.tensor(np.stack([lo, w.dec_hi])[:, ::-1].copy()).float()
        self.register_buffer("row", bank.view(2, 1, 1, -1), persistent=False)
        self.register_buffer(
            "col", bank.view(2, 1, -1, 1).repeat(2, 1, 1, 1), persistent=False
        )
        self.register_buffer(
            "y_w",
            torch.tensor([65.481, 128.553, 24.966]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "band_w",
            torch.tensor([0.035, 0.025, 0.02]).view(1, 3, 1, 1),
            persistent=False,
        )

    def swt(self, y: Tensor) -> Tensor:
        """Level-1 stationary wavelet transform of (N, 1, H, W) with periodic extension: (N, 4, H, W)
        as LL, LH, HL, HH. Rows then columns; H and W must be at least half the filter length."""
        n = self.row.shape[-1] // 2
        y = F.conv2d(F.pad(y, (n - 1, n, 0, 0), mode="circular"), self.row)  # lo, hi
        return F.conv2d(F.pad(y, (0, 0, n - 1, n), mode="circular"), self.col, groups=2)

    def forward(self, x: Tensor, it: int) -> Tensor:
        if it < self.start_iter:
            return x
        with torch.autocast(x.device.type, enabled=False):
            y = 16.0 + (x.to(self.y_w.dtype) * self.y_w).sum(
                1, keepdim=True
            )  # RGB -> Y, as BT.601 * 255
            return self.swt(y)[:, 1:4] * self.band_w  # LH, HL, HH; LL dropped
