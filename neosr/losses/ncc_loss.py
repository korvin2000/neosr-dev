from torch import Tensor, nn

from neosr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class ncc_loss(nn.Module):
    """Normalized Cross-Correlation loss.

    Args:
    ----
        loss_weight (float): weight for the loss. Default: 1.0
    """

    def __init__(self, loss_weight: float = 1.0) -> None:
        super().__init__()
        self.loss_weight = loss_weight

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
        return (1 - ((cc_value + 1) * 0.5)) * self.loss_weight
