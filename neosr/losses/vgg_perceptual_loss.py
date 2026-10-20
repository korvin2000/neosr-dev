from collections import OrderedDict
from typing import TYPE_CHECKING

from torch import Tensor, nn

from neosr.archs.vgg_arch import VGGFeatureExtractor
from neosr.losses.basic_loss import chc_loss
from neosr.utils.registry import LOSS_REGISTRY

if TYPE_CHECKING:
    from collections.abc import Callable


@LOSS_REGISTRY.register()
class vgg_perceptual_loss(nn.Module):
    """Perceptual loss with VGG19.

    Args:
    ----
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        loss_weight (float): If `loss_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        criterion (str): Criterion used for perceptual loss. Default: 'chc'
            (unclipped charbonnier).

    """

    def __init__(
        self,
        layer_weights: OrderedDict | dict[str, float] | None = None,
        vgg_type: str = "vgg19",
        use_input_norm: bool = True,
        range_norm: bool = False,
        loss_weight: float = 1.0,
        criterion: str = "chc",
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight

        if layer_weights is not None:
            self.layer_weights = layer_weights
        else:
            self.layer_weights = {
                "conv1_2": 0.1,
                "conv2_2": 0.1,
                "conv3_4": 1.0,
                "conv4_4": 1.0,
                "conv5_4": 1.0,
            }

        self.vgg = VGGFeatureExtractor(  # type: ignore[reportCallIssue]
            layer_name_list=list(self.layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm,
        )

        self.criterion: nn.L1Loss | nn.MSELoss | nn.HuberLoss | Callable
        self.criterion_type = criterion
        if self.criterion_type == "l1":
            self.criterion = nn.L1Loss()
        elif self.criterion_type == "l2":
            self.criterion = nn.MSELoss()
        elif self.criterion_type == "huber":
            self.criterion = nn.HuberLoss()
        elif self.criterion_type == "chc":
            # unclipped: feature diffs often exceed 1 and clipping zeroes their gradient
            self.criterion = chc_loss(loss_lambda=0, clip_min=0, clip_max=float("inf"))  # type: ignore[reportCallIssue]
        else:
            msg = f"{criterion} criterion not supported."
            raise NotImplementedError(msg)

    def forward(self, x: Tensor, gt: Tensor) -> Tensor:
        """Forward function.

        Args:
        ----
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
        -------
            Tensor: loss value.

        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())
        percep_loss = 0.0
        for k in x_features:
            # decrease magnitude to balance other losses
            percep_loss += (
                self.criterion(x_features[k] * 0.1, gt_features[k] * 0.1)
                * self.layer_weights[k]
            )

        return percep_loss * self.loss_weight
