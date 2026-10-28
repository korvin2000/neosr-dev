from pathlib import Path
from urllib import request

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision import models

from neosr.losses.base import Loss


class L2pooling(nn.Module):
    def __init__(
        self, filter_size: int = 5, stride: int = 2, channels: int = 64
    ) -> None:
        super().__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.tensor(a[:, None] * a[None, :], dtype=torch.float32)
        g = g / torch.sum(g)
        self.register_buffer(
            "filter", g[None, None, :, :].repeat((self.channels, 1, 1, 1))
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x**2
        out = F.conv2d(
            x, self.filter, stride=self.stride, padding=self.padding, groups=x.shape[1]
        )
        return (out + 1e-12).sqrt()


class dists_loss(Loss):
    r"""DISTS. "Image Quality Assessment: Unifying Structure and Texture Similarity":
    https://arxiv.org/abs/2004.07728.

    Args:
    ----
        as_loss (bool): True to use as loss, False for metric.
            Default: True.
        load_weights (bool): loads pretrained weights for DISTS.
            Default: True.

    """

    fp32 = True

    def __init__(
        self,
        as_loss: bool = True,
        load_weights: bool = True,
    ) -> None:
        super().__init__()
        self.as_loss = as_loss

        # only the feature stages are used: no 120M-parameter classifier is built or initialised
        with torch.device("meta"):
            vgg_pretrained_features = models.vgg16(weights=None).features
        vgg_pretrained_features.to_empty(device="cpu")
        vgg_sd = models.VGG16_Weights.DEFAULT.get_state_dict(progress=False)
        vgg_pretrained_features.load_state_dict(
            {k[9:]: v for k, v in vgg_sd.items() if k.startswith("features.")}
        )
        del vgg_sd
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])  # type: ignore[reportIndexIssue]
        self.stage2.add_module(str(4), L2pooling(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])  # type: ignore[reportIndexIssue]
        self.stage3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])  # type: ignore[reportIndexIssue]
        self.stage4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])  # type: ignore[reportIndexIssue]
        self.stage5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])  # type: ignore[reportIndexIssue]

        self.chns = [3, 64, 128, 256, 512, 512]
        self.register_parameter(
            "alpha", nn.Parameter(torch.randn(1, sum(self.chns), 1, 1))
        )
        self.register_parameter(
            "beta", nn.Parameter(torch.randn(1, sum(self.chns), 1, 1))
        )
        self.alpha.data.normal_(0.1, 0.01)
        self.beta.data.normal_(0.1, 0.01)

        # after alpha/beta: they are fixed DISTS weights, not trained
        for param in self.parameters():
            param.requires_grad = False

        # ImageNet input normalisation of the reference DISTS (alpha/beta were fitted on it)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1),
            persistent=False,
        )

        # load dists weights
        if load_weights:
            model_path = Path(__file__).parent / "dists_weights.pth"
            if not model_path.exists():
                url = "https://huggingface.co/neosr/models/resolve/main/dists_weights.pth?download=true"
                try:
                    request.urlretrieve(url, model_path)  # noqa: S310
                except OSError as err:
                    msg = "Could not download DISTS weights."
                    raise ValueError(msg) from err

            weights = torch.load(model_path, map_location="cpu", weights_only=True)
            self.alpha.data = weights["alpha"]
            self.beta.data = weights["beta"]

    def forward_once(self, x: Tensor) -> list[Tensor]:
        h = (x - self.mean) / self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        feats0 = self.forward_once(x)
        feats1 = self.forward_once(y)
        dist1 = 0
        dist2 = 0
        c1 = 1e-6
        c2 = 1e-6

        w_sum = self.alpha.sum() + self.beta.sum()
        alpha = torch.split(self.alpha / w_sum, self.chns, dim=1)
        beta = torch.split(self.beta / w_sum, self.chns, dim=1)
        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2, 3], keepdim=True)
            y_mean = feats1[k].mean([2, 3], keepdim=True)
            S1 = (2 * x_mean * y_mean + c1) / (x_mean**2 + y_mean**2 + c1)
            dist1 = dist1 + (alpha[k] * S1).sum(1, keepdim=True)

            x_var = ((feats0[k] - x_mean) ** 2).mean([2, 3], keepdim=True)
            y_var = ((feats1[k] - y_mean) ** 2).mean([2, 3], keepdim=True)
            xy_cov = (feats0[k] * feats1[k]).mean(
                [2, 3], keepdim=True
            ) - x_mean * y_mean
            S2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 = dist2 + (beta[k] * S2).sum(1, keepdim=True)

        if self.as_loss:
            out = 1 - (dist1 + dist2).mean()  # type: ignore[attr-defined]
        else:
            out = 1 - (dist1 + dist2).squeeze()  # type: ignore[attr-defined]

        return out
