from torch import Tensor

from neosr.losses.base import Loss


class ncc_loss(Loss):
    """Normalized Cross-Correlation loss."""

    def _cc(self, net_output: Tensor, gt: Tensor) -> Tensor:
        # correlate per (N, C); flatten() also handles channels_last
        a = net_output.flatten(2).float()
        b = gt.flatten(2).float()
        da = a - a.mean(2, keepdim=True)
        db = b - b.mean(2, keepdim=True)
        # clamp: a constant channel would give 0/0
        denom = ((da**2).sum(2) * (db**2).sum(2)).clamp_min(1e-8).sqrt()
        return ((da * db).sum(2) / denom).mean()

    def forward(self, net_output: Tensor, gt: Tensor) -> Tensor:
        cc_value = self._cc(net_output, gt)
        return 1 - ((cc_value + 1) * 0.5)
